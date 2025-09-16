import { $articles, $sources, inArray, asc, desc } from '@meridian/database';
import { DomainRateLimiter } from '../lib/rateLimiter'; // Verify path
import { Env } from '../index'; // Assuming Env is correctly exported from src/index.ts
import { getDb } from '../lib/utils'; // Verify path
import { parseRSSFeed } from '../lib/parsers'; // Verify path
import { WorkflowEntrypoint, WorkflowStep, WorkflowEvent, WorkflowStepConfig } from 'cloudflare:workers';
import { getRssFeedWithFetch } from '../lib/puppeteer'; // Verify path
// import { startProcessArticleWorkflow } from './processArticles.workflow'; // <<<< REMOVED: No longer directly triggering ProcessArticles
import { err, ok, ResultAsync } from 'neverthrow';


type Params = { force?: boolean };

const tierIntervals = {
    1: 60 * 60 * 1000, // Tier 1: Check every hour
    2: 4 * 60 * 60 * 1000, // Tier 2: Check every 4 hours
    3: 6 * 60 * 60 * 1000, // Tier 3: Check every 6 hours
    4: 24 * 60 * 60 * 1000, // Tier 4: Check every 24 hours
};

const dbStepConfig: WorkflowStepConfig = {
    retries: { limit: 3, delay: '1 second', backoff: 'linear' },
    timeout: '15 seconds', // Slightly increased DB timeout
};

const feedScrapeStepConfig: WorkflowStepConfig = { // Renamed for clarity
    retries: { limit: 2, delay: '3 seconds', backoff: 'exponential' }, // Fewer retries for potentially problematic feeds
    timeout: '90 seconds', // Allow more time for individual feed fetch + parse
};


// Takes in a rss feed URL, parses the feed & stores the data in our database.
export class ScrapeRssFeed extends WorkflowEntrypoint<Env, Params> {
    async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
        console.log('[ScrapeRssFeed] Workflow run instance started.');
        try {
            const db = getDb(this.env.DATABASE_URL);
            console.log('[ScrapeRssFeed] Database object initialized.');

            // Fetch all sources
            const feedsToConsider = await step.do('get_all_sources_from_db', dbStepConfig, async () => {
                console.log('[ScrapeRssFeed] Fetching all sources from DB...');
                let allFeedsFromDb = await db
                    .select({
                        id: $sources.id,
                        lastChecked: $sources.lastChecked,
                        scrape_frequency: $sources.scrape_frequency,
                        url: $sources.url,
                    })
                    .from($sources)
                    .orderBy(asc($sources.id)); // Added consistent ordering
                console.log(`[ScrapeRssFeed] Found ${allFeedsFromDb.length} total sources in DB.`);
                return allFeedsFromDb;
            });

            let dueFeeds = feedsToConsider;
            if (_event.payload.force === undefined || _event.payload.force === false) {
                console.log('[ScrapeRssFeed] Filtering feeds based on lastChecked and scrape_frequency...');
                dueFeeds = feedsToConsider.filter(feed => {
                    if (feed.lastChecked === null) {
                        console.log(`[ScrapeRssFeed] Source ID ${feed.id} (${feed.url}) never checked, including.`);
                        return true;
                    }
                    const lastCheckedTime =
                        feed.lastChecked instanceof Date ? feed.lastChecked.getTime() : new Date(feed.lastChecked).getTime();
                    const interval = tierIntervals[feed.scrape_frequency as keyof typeof tierIntervals] || tierIntervals[2];
                    const shouldCheck = Date.now() - lastCheckedTime >= interval;
                    if (shouldCheck) {
                        console.log(`[ScrapeRssFeed] Source ID ${feed.id} (${feed.url}) due for check, including.`);
                    }
                    return shouldCheck;
                });
            } else {
                console.log('[ScrapeRssFeed] Force mode enabled, including all feeds.');
            }
            console.log(`[ScrapeRssFeed] ${dueFeeds.length} feeds selected for scraping this run.`);

            // Map to the simpler structure needed by rateLimiter and downstream processing
            const feedsToProcessThisRun = dueFeeds.map(e => ({ id: e.id, url: e.url }));


            if (feedsToProcessThisRun.length === 0) {
                console.log('[ScrapeRssFeed] All feeds are up to date (or no due feeds found), exiting early...');
                return;
            }

            // Process feeds with rate limiting
            const now = Date.now();
            const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
            // const allArticlesCollected: Array<{ sourceId: number; link: string; pubDate: Date | null; title: string }> = []; // <<<< ORIGINAL
            const allRawArticles: Array<{ sourceId: number; link: string; pubDate: Date | null; title: string }> = []; // <<<< NEW: Renamed for clarity

            const rateLimiter = new DomainRateLimiter<{ id: number; url: string }>({
                maxConcurrent: 5, // Reduced concurrency
                globalCooldownMs: 1000,
                domainCooldownMs: 3000,
            });

            console.log(`[ScrapeRssFeed] Starting batch processing for ${feedsToProcessThisRun.length} selected feeds.`);
            const feedResults = await rateLimiter.processBatch(feedsToProcessThisRun, step, async (feed, _domain) => {
                console.log(`[ScrapeRssFeed] [RateLimiter] Processing feed ID: ${feed.id}, URL: ${feed.url}`);
                try {
                    // Use the feedScrapeStepConfig here
                    return await step.do(
                        `scrape_and_parse_feed_${feed.id}`, // More descriptive step name
                        feedScrapeStepConfig, // Using the config with longer timeout
                        async () => {
                            console.log(`[ScrapeRssFeed] Attempting to fetch feed ID: ${feed.id}, URL: ${feed.url}`);
                            const feedPage = await getRssFeedWithFetch(feed.url); // This can be slow
                            if (feedPage.isErr()) {
                                console.error(`[ScrapeRssFeed] Error fetching feed ID ${feed.id} (${feed.url}): ${feedPage.error.type} - ${feedPage.error.message || 'No specific message'}`);
                                throw feedPage.error; // Let step.do handle retries based on feedScrapeStepConfig
                            }
                            console.log(`[ScrapeRssFeed] Successfully fetched feed ID: ${feed.id}. Parsing...`);

                            const feedArticles = await parseRSSFeed(feedPage.value); // This can also be slow or error
                            if (feedArticles.isErr()) {
                                console.error(`[ScrapeRssFeed] Error parsing feed ID ${feed.id} (${feed.url}): ${feedArticles.error.type} - ${feedArticles.error.message || 'No specific message'}`);
                                throw feedArticles.error; // Let step.do handle retries
                            }
                            console.log(`[ScrapeRssFeed] Successfully parsed feed ID: ${feed.id}. Found ${feedArticles.value.length} raw articles.`);

                            const filteredArticles = feedArticles.value.filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo);
                            console.log(`[ScrapeRssFeed] Feed ID: ${feed.id}. Filtered to ${filteredArticles.length} articles (newer than 1 week).`);
                            return filteredArticles.map(e => ({ ...e, sourceId: feed.id }));
                        }
                    );
                } catch (error: any) { // Catch errors from the step.do block (after retries failed)
                    const errorMessage = error?.message || String(error);
                    console.error(`[ScrapeRssFeed] [RateLimiter] Critical failure after retries for feed ID ${feed.id} (${feed.url}): ${errorMessage}`);
                    return []; // Return empty array for this feed to not break the whole batch
                }
            });

            console.log('[ScrapeRssFeed] Batch processing of feeds complete.');
            feedResults.forEach(articles => {
                if (articles && Array.isArray(articles)) {
                    // allArticlesCollected.push(...articles); // <<<< ORIGINAL
                    allRawArticles.push(...articles); // <<<< NEW: Using the renamed array
                }
            });
            console.log(`[ScrapeRssFeed] Total raw articles collected from all feeds: ${allRawArticles.length}`); // <<<< NEW: Log updated


            // <<<< NEW BLOCK: Insert articles and publish to queue
            let newlyInsertedArticleIds: number[] = [];
            if (allRawArticles.length > 0) {
                console.log(`[ScrapeRssFeed] Attempting to insert ${allRawArticles.length} new articles into DB and queue them.`);
                newlyInsertedArticleIds = await step.do('insert_new_articles_batch', dbStepConfig, async () =>
                    db
                        .insert($articles)
                        .values(
                            allRawArticles.map(({ sourceId, link, pubDate, title }) => ({
                                sourceId, url: link, title, publishDate: pubDate,
                                // content, summary, relevance, etc. will be NULL by default as per schema
                            }))
                        )
                        .onConflictDoNothing()
                        .returning({ id: $articles.id }) // Return IDs of actually inserted rows
                ).then(rows => rows.map(row => row.id)); // Extract just the IDs
                console.log(`[ScrapeRssFeed] Successfully inserted ${newlyInsertedArticleIds.length} unique new articles.`);

                for (const articleId of newlyInsertedArticleIds) {
                    await this.env.ARTICLE_CONTENT_FETCH_QUEUE.send({ articleId });
                    console.log(`[ScrapeRssFeed] Published new article ID ${articleId} to content fetch queue.`);
                }
            } else {
                console.log('[ScrapeRssFeed] No new articles found from any feed to insert or queue in this run.');
            }
            // <<<< END NEW BLOCK

            // Original logic for updating sources, adjusted to use allRawArticles if needed
            if (allRawArticles.length > 0) { // <<<< NEW: Using allRawArticles
                const updatedSourceIds = Array.from(new Set(allRawArticles.map(({ sourceId }) => sourceId))); // <<<< NEW: Using allRawArticles
                if (updatedSourceIds.length > 0) {
                    console.log(`[ScrapeRssFeed] Attempting to update lastChecked for ${updatedSourceIds.length} sources that yielded articles.`);
                    await step.do('update_sources_with_articles_yielded', dbStepConfig, async () =>
                        db
                            .update($sources)
                            .set({ lastChecked: new Date() })
                            .where(inArray($sources.id, updatedSourceIds))
                    );
                    console.log('[ScrapeRssFeed] lastChecked update for sources with articles complete.');
                }
            } else {
                // No change needed here, original logic handles it.
            }

            // Always update lastChecked for ALL feeds that were ATTEMPTED and SELECTED for this run
            if (feedsToProcessThisRun.length > 0) {
                console.log(`[ScrapeRssFeed] Attempting to update lastChecked for all ${feedsToProcessThisRun.length} attempted feeds in this run.`);
                await step.do('update_all_attempted_feeds_in_run', dbStepConfig, async () =>
                    db
                        .update($sources)
                        .set({ lastChecked: new Date() })
                        .where(inArray($sources.id, feedsToProcessThisRun.map(feed => feed.id)))
                );
                console.log('[ScrapeRssFeed] lastChecked update for all attempted feeds in this run complete.');
            }

            // <<<< REMOVED: No longer directly triggering ProcessArticles
            // if (allArticlesCollected.length > 0) {
            //     console.log('[ScrapeRssFeed] Attempting to trigger PROCESS_ARTICLES workflow as new articles were collected...');
            //     await step.do('trigger_article_processor_workflow', dbStepConfig, async () => {
            //         const workflow = await startProcessArticleWorkflow(this.env); // Uses this.env
            //         if (workflow.isErr()) {
            //             console.error('[ScrapeRssFeed] Error starting PROCESS_ARTICLES workflow:', workflow.error.message);
            //             throw workflow.error;
            //         }
            //         console.log(`[ScrapeRssFeed] PROCESS_ARTICLES workflow started with ID: ${workflow.value.id}`);
            //         return workflow.value.id;
            //     });
            // } else {
            //     console.log('[ScrapeRssFeed] No new articles collected, so PROCESS_ARTICLES workflow will not be triggered.');
            // }
            // <<<< END REMOVED BLOCK

            console.log('[ScrapeRssFeed] Workflow run finished successfully.');

        } catch (error: any) {
            const errorMessage = error?.message || String(error);
            const errorStack = error?.stack || 'No stack available';
            console.error(`[ScrapeRssFeed] CRITICAL UNHANDLED ERROR in workflow run: ${errorMessage}`);
            console.error(`[ScrapeRssFeed] Error Stack: ${errorStack}`);
            // For Cloudflare logs, sometimes serializing parts of the error is useful
            if (typeof error === 'object' && error !== null) {
                console.error('[ScrapeRssFeed] Full error object properties:', JSON.stringify(error, Object.getOwnPropertyNames(error)));
            }
            throw error; // Re-throw to ensure the workflow invocation is marked as failed if not caught by step.do
        }
    }
}

export async function startRssFeedScraperWorkflow(env: Env, params?: Params) {
    console.log("[Workflow Starter] Attempting to start ScrapeRssFeed workflow. Params:", params); // Log params
    const workflowParams = params || {}; // Ensure params is an object
    const workflow = await ResultAsync.fromPromise(
        env.SCRAPE_RSS_FEED.create({ id: crypto.randomUUID(), params: workflowParams }),
        e => e instanceof Error ? e : new Error(String(e))
    );
    if (workflow.isErr()) {
        console.error("[Workflow Starter] Error creating SCRAPE_RSS_FEED workflow instance:", workflow.error.message);
        return err(workflow.error);
    }
    console.log(`[Workflow Starter] SCRAPE_RSS_FEED workflow instance created successfully with ID: ${workflow.value.id}`);
    return ok(workflow.value);
}