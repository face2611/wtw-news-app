// C:\Users\phili\meridian\apps\scrapers\src\logic\rssFeed.logic.ts
// --- Full & Correct Scraper Logic (postgres-js client) ---

// --- Drizzle DB Client Setup (SELF-CONTAINED) ---
import { drizzle } from 'drizzle-orm/postgres-js'; // Use postgres-js adapter
import postgres from 'postgres'; // Use postgres client
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { eq, inArray, asc, desc } from 'drizzle-orm';

// --- Define schema INLINE for this worker ---
export const $sources = pgTable('sources', {
  id: serial('id').primaryKey(),
  url: text('url').notNull().unique(),
  name: text('name').notNull(),
  scrape_frequency: integer('scrape_frequency').notNull().default(2),
  paywall: boolean('paywall').notNull().default(false),
  category: text('category').notNull(),
  lastChecked: timestamp('last_checked', { mode: 'date' }),
});

export const $articles = pgTable('articles', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  url: text('url').notNull().unique(),
  publishDate: timestamp('publish_date', { mode: 'date' }),
  content: text('content'),
  processing_status: text('processing_status').default('Scraped'),
  contentFetchedAt: timestamp('content_fetched_at', { mode: 'date' }),
  geminiProcessedAt: timestamp('gemini_processed_at', { mode: 'date' }),
  run_id: text('run_id'),
  language: text('language'),
  location: text('location'),
  completeness: text('completeness'),
  relevance: text('relevance'),
  summary: text('summary'),
  failReason: text('fail_reason'),
  sourceId: integer('source_id')
    .references(() => $sources.id)
    .notNull(),
  processedAt: timestamp('processed_at', { mode: 'date' }),
  createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});
// --- End INLINE SCHEMA ---


let dbClient = null; // Declare dbClient here for lazy init

function getDb(databaseUrl) { // This accepts the DATABASE_URL secret (a string)
    if (!dbClient) {
        const queryClient = postgres(databaseUrl); // Use postgres client with string
        dbClient = drizzle(queryClient, { schema: {
            articles: $articles,
            sources: $sources,
        }});
    }
    return dbClient;
}

// --- End Drizzle DB Client Setup ---


// --- Helper function for timestamp formatting ---
function formatTimestampForPgWithoutTimeZone(date) {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}
// --- End Helper ---


// --- DomainRateLimiter class definition (moved here for isolation) ---
// This was causing 'not defined' errors when imported from another file or not bundled correctly.
// Defining it inline ensures it's always part of this worker's code.
class DomainRateLimiter<T extends { id: number; url: string }> {
    private maxConcurrent: number;
    private globalCooldownMs: number;
    private domainCooldownMs: number;
    private queue: Array<{ item: T; resolve: (value: any) => void; reject: (reason?: any) => void; domain: string; processor: (item: T, domain: string) => Promise<any>; }>;
    private activeRequests: number;
    private domainCooldowns: Map<string, number>;
    private globalCooldownTimer: Promise<void> | null; // Keep null as default

    constructor(options: { maxConcurrent: number; globalCooldownMs: number; domainCooldownMs: number }) {
        this.maxConcurrent = options.maxConcurrent;
        this.globalCooldownMs = options.globalCooldownMs;
        this.domainCooldownMs = options.domainCooldownMs;
        this.queue = [];
        this.activeRequests = 0;
        this.domainCooldowns = new Map();
        this.globalCooldownTimer = null;
    }

    async processBatch(items: T[], processor: (item: T, domain: string) => Promise<any>): Promise<any[]> {
        const results: Promise<any>[] = [];

        for (const item of items) {
            const domain = new URL(item.url).hostname;
            results.push(new Promise((resolve, reject) => {
                this.queue.push({ item, resolve, reject, domain, processor });
                this.processNext();
            }));
        }

        return Promise.allSettled(results.map(p => p.catch(e => e)));
    }

    private async processNext() {
        if (this.activeRequests >= this.maxConcurrent || this.queue.length === 0) {
            return;
        }

        const now = Date.now();
        let nextItemIndex = -1;
        for (let i = 0; i < this.queue.length; i++) {
            const { domain } = this.queue[i];
            const lastUsed = this.domainCooldowns.get(domain) || 0;

            if (now - lastUsed >= this.domainCooldownMs) {
                // Simplified global cooldown check for this context
                // Original logic here was more complex but let's try this simple version first
                if (!this.globalCooldownTimer || this.globalCooldownTimer === null) {
                    nextItemIndex = i;
                    break;
                }
            }
        }
        if (nextItemIndex === -1) {
            setTimeout(() => this.processNext(), 100);
            return;
        }

        const { item, resolve, reject, domain, processor } = this.queue.splice(nextItemIndex, 1)[0];
        this.activeRequests++;
        this.domainCooldowns.set(domain, now);

        // Start global cooldown timer if not active
        if (!this.globalCooldownTimer) {
            this.globalCooldownTimer = new Promise(res => setTimeout(res, this.globalCooldownMs));
        }

        try {
            const result = await processor(item, domain); // Execute stored processor
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.activeRequests--;
            this.globalCooldownTimer = null; // Reset global cooldown after one item for simplicity
            setTimeout(() => this.processNext(), 0); // Continue processing immediately
        }
    }
}
// --- END DomainRateLimiter class definition ---


// --- Existing imports for other services ---
import { Env } from '../index'; // Assuming Env is correctly exported from src/index.ts
import { parseRSSFeed } from '../lib/parsers';
import { getRssFeedWithFetch } from '../lib/puppeteer';
import { err, ok, ResultAsync } from 'neverthrow';
// --- End existing imports ---


const tierIntervals = {
    1: 60 * 60 * 1000, // Tier 1: Check every hour
    2: 4 * 60 * 60 * 1000, // Tier 2: Check every 4 hours
    3: 6 * 60 * 60 * 1000, // Tier 3: Check every 6 hours
    4: 24 * 60 * 60 * 1000, // Tier 4: Check every 24 hours
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
            db = getDb(env.DATABASE_URL); // Pass env.DATABASE_URL (string)
        } catch (dbError) {
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
        } catch (sourcesError) {
            throw new Error(`Fetch Sources Failed. Run ID: ${currentRunId}: ${sourcesError?.message || String(sourcesError)}`);
        }

        if (allFeedsFromDb.length === 0) {
            console.error(`[ScrapeRssFeed] DEBUG: No sources found in DB. Exiting. Run ID: ${currentRunId}`);
            return;
        }

        // --- Original filtering logic (no hard limit here) ---
        let dueFeeds = allFeedsFromDb;
        if (params.force === undefined || params.force === false) {
             dueFeeds = allFeedsFromDb.filter(feed => {
                if (feed.lastChecked === null) return true;
                const lastCheckedTime =
                    feed.lastChecked instanceof Date ? feed.lastChecked.getTime() : new Date(feed.lastChecked).getTime();
                const interval = tierIntervals[feed.scrape_frequency] || tierIntervals[2];
                return Date.now() - lastCheckedTime >= interval;
            });
        }
        // --- END Original filtering ---

        if (dueFeeds.length === 0) {
            console.error(`[ScrapeRssFeed] DEBUG: No feeds selected for scraping after filtering. Exiting. Run ID: ${currentRunId}`);
            return;
        }

        const feedsToProcessThisRun = dueFeeds.map(e => ({ id: e.id, url: e.url }));
        console.error(`[ScrapeRssFeed] DEBUG: Actual feeds to process this run: ${feedsToProcessThisRun.length}. Run ID: ${currentRunId}`);

        const now = Date.now();
        const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
        const allRawArticles = [];

        const rateLimiter = new DomainRateLimiter({ // Use the inline class
            maxConcurrent: 5,
            globalCooldownMs: 1000,
            domainCooldownMs: 3000,
        });

        let feedResults;
        try {
            // Pass the processor function directly to processBatch
            feedResults = await rateLimiter.processBatch(feedsToProcessThisRun, async (feed, _domain) => {
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
                } catch (error) {
                    console.error(`[ScrapeRssFeed] ERROR: Internal batch processing for feed ID ${feed.id}: ${error?.message || String(error)}. Run ID: ${currentRunId}`);
                    return [];
                }
            });
        } catch (batchError) {
            throw new Error(`Batch Processing Failed. Run ID: ${currentRunId}: ${batchError?.message || String(batchError)}`);
        }

                feedResults.forEach(result => {
            if (result.status === 'fulfilled' && Array.isArray(result.value)) {
                allRawArticles.push(...result.value);
            } else if (result.status === 'rejected') {
                // Log the error to understand why a specific feed failed
                console.error(`[ScrapeRssFeed] ERROR: Failed to process a feed batch item: ${result.reason}. Run ID: ${currentRunId}`);
            }
        });
        console.error(`[ScrapeRssFeed] DEBUG: Total raw articles collected: ${allRawArticles.length}. Run ID: ${currentRunId}`);

        let newlyInsertedArticleIds = [];
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

            } catch (insertError) {
                throw new Error(`Article Insert Failed. Run ID: ${currentRunId}: ${insertError?.message || String(insertError)}`);
            }

            if (newlyInsertedArticleIds.length > 0) {
                try {
                    console.error(`[ScrapeRssFeed] DEBUG: Attempting to publish ${newlyInsertedArticleIds.length} articles to queue. Run ID: ${currentRunId}`);
                    for (const articleId of newlyInsertedArticleIds) {
                        await env.ARTICLE_CONTENT_FETCH_QUEUE.send({ articleId, runId: currentRunId });
                        await db.update($articles).set({ processing_status: 'Queued_For_Content_Fetch' })
                                .where(eq($articles.id, articleId));
                    }
                    console.error(`[ScrapeRssFeed] DEBUG: Successfully queued and updated status for ${newlyInsertedArticleIds.length} articles. Run ID: ${currentRunId}`);
                } catch (queueError) {
                    await db.update($articles).set({ processing_status: 'Queue_Publish_Failed' })
                            .where(inArray($articles.id, newlyInsertedArticleIds));
                    throw new Error(`Queue Publish Failed. Run ID: ${currentRunId}: ${queueError?.message || String(queueError)}. Article ID: ${newlyInsertedArticleIds.join(',')}`);
                }
            }
        } else {
            console.error(`[ScrapeRssFeed] DEBUG: No new articles found from any feed to insert or queue in this run. Run ID: ${currentRunId}`);
        }

        if (feedsToProcessThisRun.length > 0) {
            try {
                await db
                    .update($sources)
                    .set({ lastChecked: new Date() })
                    .where(inArray($sources.id, feedsToProcessThisRun.map(feed => feed.id)));
                console.error(`[ScrapeRssFeed] DEBUG: Updated lastChecked for processed source(s). Run ID: ${currentRunId}`);
            } catch (updateError) {
                console.error(`[ScrapeRssFeed] Source Update Error: ${updateError?.message || String(updateError)}. Run ID: ${currentRunId}`);
            }
        } else {
            console.error(`[ScrapeRssFeed] DEBUG: No feeds selected for processing, skipping lastChecked update for all attempted feeds. Run ID: ${currentRunId}`);
        }

        console.error(`[ScrapeRssFeed] DEBUG: Function run finished successfully. Run ID: ${currentRunId}`);

    } catch (topLevelError) {
        throw new Error(`Top-level ScrapeRssFeed Error. Run ID: ${currentRunId}: ${topLevelError?.message || String(topLevelError)}`);
    }
}