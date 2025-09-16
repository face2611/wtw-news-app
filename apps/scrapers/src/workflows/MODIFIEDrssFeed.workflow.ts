import { $articles, $sources, inArray } from '@meridian/database';
import { DomainRateLimiter } from '../lib/rateLimiter';
import { Env } from '../index'; // Assuming Env is correctly exported from src/index.ts
import { getDb } from '../lib/utils';
import { parseRSSFeed } from '../lib/parsers';
import { WorkflowEntrypoint, WorkflowStep, WorkflowEvent, WorkflowStepConfig } from 'cloudflare:workers';
import { getRssFeedWithFetch } from '../lib/puppeteer';
import { startProcessArticleWorkflow } from './processArticles.workflow'; // Assuming this path is correct
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
  timeout: '5 seconds',
};

// Takes in a rss feed URL, parses the feed & stores the data in our database.
export class ScrapeRssFeed extends WorkflowEntrypoint<Env, Params> {
  async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
    console.log('[ScrapeRssFeed] Workflow run started.'); // LOG: Workflow start
    const db = getDb(this.env.DATABASE_URL);

    // Fetch all sources
    const feeds = await step.do('get feeds', dbStepConfig, async () => {
      console.log('[ScrapeRssFeed] Fetching all sources from DB...'); // LOG
      let allFeedsFromDb = await db
        .select({
          id: $sources.id,
          lastChecked: $sources.lastChecked,
          scrape_frequency: $sources.scrape_frequency,
          url: $sources.url,
        })
        .from($sources);
      console.log(`[ScrapeRssFeed] Found ${allFeedsFromDb.length} total sources in DB.`); // LOG

      if (_event.payload.force === undefined || _event.payload.force === false) {
        console.log('[ScrapeRssFeed] Filtering feeds based on lastChecked and scrape_frequency...'); // LOG
        allFeedsFromDb = allFeedsFromDb.filter(feed => {
          if (feed.lastChecked === null) {
            console.log(`[ScrapeRssFeed] Source ID ${feed.id} never checked, including.`); // LOG
            return true;
          }
          const lastCheckedTime =
            feed.lastChecked instanceof Date ? feed.lastChecked.getTime() : new Date(feed.lastChecked).getTime();
          const interval = tierIntervals[feed.scrape_frequency as keyof typeof tierIntervals] || tierIntervals[2];
          const shouldCheck = Date.now() - lastCheckedTime >= interval;
          if (shouldCheck) {
            console.log(`[ScrapeRssFeed] Source ID ${feed.id} due for check, including.`); // LOG
          }
          return shouldCheck;
        });
      } else {
        console.log('[ScrapeRssFeed] Force mode enabled, including all feeds.'); // LOG
      }
      console.log(`[ScrapeRssFeed] ${allFeedsFromDb.length} feeds selected for scraping.`); // LOG
      return allFeedsFromDb.map(e => ({ id: e.id, url: e.url }));
    });

    if (feeds.length === 0) {
      console.log('[ScrapeRssFeed] All feeds are up to date, exiting early...');
      return;
    }

    // Process feeds with rate limiting
    const now = Date.now();
    const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
    const allArticles: Array<{ sourceId: number; link: string; pubDate: Date | null; title: string }> = [];

    // Create rate limiter
    const rateLimiter = new DomainRateLimiter<{ id: number; url: string }>({
      maxConcurrent: 10,
      globalCooldownMs: 500,
      domainCooldownMs: 2000,
    });

    console.log(`[ScrapeRssFeed] Starting batch processing for ${feeds.length} feeds with rate limiter.`); // LOG
    // Process feeds with rate limiting
    const feedResults = await rateLimiter.processBatch(feeds, step, async (feed, _domain) => {
      console.log(`[ScrapeRssFeed] [RateLimiter] Processing feed ID: ${feed.id}, URL: ${feed.url}`); // LOG
      try {
        return await step.do(
          `scrape feed ${feed.id}`,
          {
            retries: { limit: 3, delay: '2 seconds', backoff: 'exponential' },
          },
          async () => {
            console.log(`[ScrapeRssFeed] Attempting to fetch feed ID: ${feed.id}, URL: ${feed.url}`); // LOG
            const feedPage = await getRssFeedWithFetch(feed.url);
            if (feedPage.isErr()) {
              console.error(`[ScrapeRssFeed] Error fetching feed ID ${feed.id}, URL ${feed.url}: ${feedPage.error.type} - ${feedPage.error.message}`); // LOG ERROR
              throw feedPage.error; // This will be caught by the outer catch
            }
            console.log(`[ScrapeRssFeed] Successfully fetched feed ID: ${feed.id}. Parsing...`); // LOG

            const feedArticles = await parseRSSFeed(feedPage.value);
            if (feedArticles.isErr()) {
              console.error(`[ScrapeRssFeed] Error parsing feed ID ${feed.id}, URL ${feed.url}: ${feedArticles.error.type} - ${feedArticles.error.message}`); // LOG ERROR
              throw feedArticles.error; // This will be caught by the outer catch
            }
            console.log(`[ScrapeRssFeed] Successfully parsed feed ID: ${feed.id}. Found ${feedArticles.value.length} raw articles.`); // LOG

            // Filter articles older than one week
            const filteredArticles = feedArticles.value.filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo);
            console.log(`[ScrapeRssFeed] Feed ID: ${feed.id}. Filtered to ${filteredArticles.length} articles (newer than 1 week).`); // LOG
            return filteredArticles.map(e => ({ ...e, sourceId: feed.id }));
          }
        );
      } catch (error: any) { // Catch errors from the step.do block
        // Ensure error has a message property or convert to string
        const errorMessage = error?.message || String(error);
        console.error(`[ScrapeRssFeed] [RateLimiter] Critical error processing feed ID ${feed.id}: ${errorMessage}`); // LOG ERROR
        return []; // Return empty array for this feed on critical error to not break batch
      }
    });

    console.log('[ScrapeRssFeed] Batch processing of feeds complete.'); // LOG
    // Flatten the results into allArticles
    feedResults.forEach(articles => {
      if (articles && Array.isArray(articles)) { // Ensure articles is an array
        allArticles.push(...articles);
      }
    });
    console.log(`[ScrapeRssFeed] Total new articles collected from all feeds: ${allArticles.length}`); // LOG

    // Insert articles and update sources
    if (allArticles.length > 0) {
      console.log(`[ScrapeRssFeed] Attempting to insert ${allArticles.length} new articles into DB.`); // LOG
      await step.do('insert new articles', dbStepConfig, async () =>
        db
          .insert($articles)
          .values(
            allArticles.map(({ sourceId, link, pubDate, title }) => ({
              sourceId,
              url: link,
              title,
              publishDate: pubDate,
              // Let other fields like content, summary, relevance, location, processed_at be NULL initially
            }))
          )
          .onConflictDoNothing() // Important for idempotency
      );
      console.log('[ScrapeRssFeed] Article insertion step complete.'); // LOG

      const updatedSourceIds = Array.from(new Set(allArticles.map(({ sourceId }) => sourceId)));
      console.log(`[ScrapeRssFeed] Attempting to update lastChecked for ${updatedSourceIds.length} sources that yielded articles.`); // LOG
      await step.do('update sources that yielded articles', dbStepConfig, async () => // Changed step name
        db
          .update($sources)
          .set({ lastChecked: new Date() })
          .where(inArray($sources.id, updatedSourceIds))
      );
      console.log('[ScrapeRssFeed] lastChecked update for sources with articles complete.'); // LOG
    } else {
      console.log('[ScrapeRssFeed] No new articles found from any feed.'); // LOG
    }

    // Always update lastChecked for ALL feeds that were ATTEMPTED in this run
    // This ensures feeds that returned 0 new articles but were successfully checked get updated
    const attemptedFeedIds = feeds.map(feed => feed.id);
    if (attemptedFeedIds.length > 0) {
      console.log(`[ScrapeRssFeed] Attempting to update lastChecked for all ${attemptedFeedIds.length} attempted feeds.`); // LOG
      await step.do('update all attempted sources', dbStepConfig, async () => // Changed step name
        db
          .update($sources)
          .set({ lastChecked: new Date() })
          .where(inArray($sources.id, attemptedFeedIds))
      );
      console.log('[ScrapeRssFeed] lastChecked update for all attempted feeds complete.'); // LOG
    }


    console.log('[ScrapeRssFeed] Attempting to trigger PROCESS_ARTICLES workflow...'); // LOG
    await step.do('trigger_article_processor', dbStepConfig, async () => {
      const workflow = await startProcessArticleWorkflow(this.env);
      if (workflow.isErr()) {
        console.error('[ScrapeRssFeed] Error starting PROCESS_ARTICLES workflow:', workflow.error); // LOG ERROR
        throw workflow.error;
      }
      console.log(`[ScrapeRssFeed] PROCESS_ARTICLES workflow started with ID: ${workflow.value.id}`); // LOG
      return workflow.value.id;
    });

    console.log('[ScrapeRssFeed] Workflow run finished.'); // LOG: Workflow end
  }
}

export async function startRssFeedScraperWorkflow(env: Env, params?: Params) {
  // Add logging here if needed, e.g., console.log("startRssFeedScraperWorkflow called with params:", params);
  const workflow = await ResultAsync.fromPromise(
    env.SCRAPE_RSS_FEED.create({ id: crypto.randomUUID(), params: params || {} }), // Ensure params is not undefined
    e => e instanceof Error ? e : new Error(String(e))
  );
  if (workflow.isErr()) {
    // Add logging here too, e.g., console.error("Error creating SCRAPE_RSS_FEED workflow instance:", workflow.error);
    return err(workflow.error);
  }
  return ok(workflow.value);
}