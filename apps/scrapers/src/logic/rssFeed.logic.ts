// C:\Users\phili\meridian\apps\scrapers\src\logic\rssFeed.logic.ts

// --- NEW: Import Drizzle related modules from the new local db.js file ---
import { getDb, $articles, $sources } from '../db.js'; // <<<< CORRECTED: Imports getDb, $articles, $sources from your new local db.js
// --- END NEW IMPORTS ---

import { DomainRateLimiter } from '../lib/rateLimiter';
import { Env } from '../index';
import { parseRSSFeed } from '../lib/parsers';
import { getRssFeedWithFetch } from '../lib/puppeteer';
import { inArray, asc, desc, eq } from 'drizzle-orm'; // <<<< Keep individual Drizzle ops imported from drizzle-orm itself

// REMOVED imports:
// import { drizzle } from 'drizzle-orm/postgres-js';
// import postgres from 'postgres';
// import * as schema from '../../../packages/database/src/schema';
// function getDb(databaseUrl: string) { ... } // Removed since it's now in db.js


const tierIntervals = {
    1: 60 * 60 * 1000,
    2: 4 * 60 * 60 * 1000,
    3: 6 * 60 * 60 * 1000,
    4: 24 * 60 * 60 * 1000,
};

export async function runScrapeRssFeedLogic(env: Env, ctx: ExecutionContext, params: { force?: boolean }) {
    const currentRunId = crypto.randomUUID();
    console.error(`[ScrapeRssFeed] DEBUG: Starting run ${currentRunId}`);

    try {
        if (!env.ARTICLE_CONTENT_FETCH_QUEUE) {
            throw new Error(`[ScrapeRssFeed] ERROR: ARTICLE_CONTENT_FETCH_QUEUE binding is missing or undefined! Run ID: ${currentRunId}`);
        }

        let db;
        try {
            db = getDb(env.DATABASE_URL); // getDb is now correctly imported from db.js
        } catch (dbError: any) {
            throw new Error(`DB Init Failed. Run ID: ${currentRunId}: ${dbError?.message || String(dbError)}`);
        }

        let allFeedsFromDb;
        try {
            allFeedsFromDb = await db
                .select({
                    id: $sources.id,
                    lastChecked: $sources.lastChecked,
                    scrape_frequency: $sources.scrape_frequency,
                    url: $sources.url,
                })
                .from($sources)
                .orderBy(asc($sources.id));
        } catch (sourcesError: any) {
            throw new Error(`Fetch Sources Failed. Run ID: ${currentRunId}: ${sourcesError?.message || String(sourcesError)}`);
        }

        // --- NEW: ABSOLUTE HARD LIMIT TO FIRST SOURCE ---
        const feedsToProcessThisRun = allFeedsFromDb.length > 0 ? [allFeedsFromDb[0]] : [];
        if (allFeedsFromDb.length > 0) {
            console.error(`[ScrapeRssFeed] DEBUG: Limiting to first source. ID: ${feedsToProcessThisRun[0].id}, URL: ${feedsToProcessThisRun[0].url}. Run ID: ${currentRunId}`);
        } else {
            console.error(`[ScrapeRssFeed] DEBUG: No sources found in DB. Exiting. Run ID: ${currentRunId}`);
            return;
        }
        // --- END NEW ---

        const now = Date.now();
        const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
        const allRawArticles: Array<{ sourceId: number; link: string; pubDate: Date | null; title: string }> = [];

        const rateLimiter = new DomainRateLimiter<{ id: number; url: string }>({
            maxConcurrent: 5,
            globalCooldownMs: 1000,
            domainCooldownMs: 3000,
        });

        let feedResults;
        try {
            feedResults = await rateLimiter.processBatch(feedsToProcessThisRun, null as any, async (feed, _domain) => {
                try {
                    const feedPage = await getRssFeedWithFetch(feed.url);
                    if (feedPage.isErr()) {
                        console.error(`[ScrapeRssFeed] ERROR: Failed to fetch RSS feed ${feed.url}: ${feedPage.error.message}. Run ID: ${currentRunId}`);
                        return [];
                    }
                    const feedArticles = await parseRSSFeed(feedPage.value);
                    if (feedArticles.isErr()) {
                        console.error(`[ScrapeRssFeed] ERROR: Failed to parse RSS feed ${feed.url}: ${feedArticles.error.message}. Run ID: ${currentRunId}`);
                        return [];
                    }
                    const filteredArticles = feedArticles.value.filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo);
                    return filteredArticles.map(e => ({ ...e, sourceId: feed.id }));
                } catch (error: any) {
                    console.error(`[ScrapeRssFeed] ERROR: Internal batch processing for feed ID ${feed.id}: ${error?.message || String(error)}. Run ID: ${currentRunId}`);
                    return [];
                }
            });
        } catch (batchError: any) {
            throw new Error(`Batch Processing Failed. Run ID: ${currentRunId}: ${batchError?.message || String(batchError)}`);
        }

        feedResults.forEach(articles => {
            if (articles && Array.isArray(articles)) {
                allRawArticles.push(...articles);
            }
        });

        console.error(`[ScrapeRssFeed] DEBUG: Total raw articles collected: ${allRawArticles.length}. Run ID: ${currentRunId}`);

        let newlyInsertedArticleIds: number[] = [];
        if (allRawArticles.length > 0) {
            try {
                const articlesToInsert = allRawArticles.map(({ sourceId, link, pubDate, title }) => ({
                    sourceId, url: link, title, publishDate: pubDate,
                    processing_status: 'Scraped',
                    run_id: currentRunId
                }));

                newlyInsertedArticleIds = await db
                    .insert($articles)
                    .values(articlesToInsert)
                    .onConflictDoNothing()
                    .returning({ id: $articles.id })
                .then(rows => rows.map(row => row.id));

                if (newlyInsertedArticleIds.length > 0) {
                    console.error(`[ScrapeRssFeed] DEBUG: Inserted ${newlyInsertedArticleIds.length} unique new articles. Proceeding to queue. Run ID: ${currentRunId}`);
                } else {
                    console.error(`[ScrapeRssFeed] DEBUG: No unique new articles were inserted. Skipping queueing. Run ID: ${currentRunId}`);
                }

            } catch (insertError: any) {
                throw new Error(`Article Insert Failed. Run ID: ${currentRunId}: ${insertError?.message || String(insertError)}`);
            }

            if (newlyInsertedArticleIds.length > 0) {
                try {
                    console.error(`[ScrapeRssFeed] DEBUG: Attempting to publish ${newlyInsertedArticleIds.length} articles to queue. Run ID: ${currentRunId}`);
                    for (const articleId of newlyInsertedArticleIds) {
                        await env.ARTICLE_CONTENT_FETCH_QUEUE.send({ articleId });
                        await db.update($articles).set({ processing_status: 'Queued_For_Content_Fetch' })
                                .where(eq($articles.id, articleId));
                    }
                    console.error(`[ScrapeRssFeed] DEBUG: Successfully queued and updated status for ${newlyInsertedArticleIds.length} articles. Run ID: ${currentRunId}`);
                } catch (queueError: any) {
                    await db.update($articles).set({ processing_status: 'Queue_Publish_Failed' })
                            .where(inArray($articles.id, newlyInsertedArticleIds));
                    throw new Error(`Queue Publish Failed. Run ID: ${currentRunId}: ${queueError?.message || String(queueError)}`);
                }
            }
        } else {
            console.error(`[ScrapeRssFeed] DEBUG: No new articles found from any feed to insert or queue in this run. Run ID: ${currentRunId}`);
        }

        // --- Source lastChecked updates - ensure currentRunId is logged on errors here too if possible ---
        if (feedsToProcessThisRun.length > 0) {
            try {
                await db
                    .update($sources)
                    .set({ lastChecked: new Date() })
                    .where(inArray($sources.id, feedsToProcessThisRun.map(feed => feed.id)));
                console.error(`[ScrapeRssFeed] DEBUG: Updated lastChecked for processed source(s). Run ID: ${currentRunId}`);
            } catch (updateError: any) {
                console.error(`[ScrapeRssFeed] Source Update Error: ${updateError?.message || String(updateError)}. Run ID: ${currentRunId}`);
            }
        } else {
            console.error(`[ScrapeRssFeed] DEBUG: No feeds selected for processing, skipping lastChecked update for all attempted feeds. Run ID: ${currentRunId}`);
        }

        console.error(`[ScrapeRssFeed] DEBUG: Function run finished successfully. Run ID: ${currentRunId}`);

    } catch (topLevelError: any) {
        throw new Error(`Top-level ScrapeRssFeed Error. Run ID: ${currentRunId}: ${topLevelError?.message || String(topLevelError)}`);
    }
}