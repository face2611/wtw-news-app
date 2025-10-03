// C:\Users\phili\meridian\apps\content-fetcher-worker\src\index.js

// --- Drizzle DB Client Setup (SELF-CONTAINED) ---
import { drizzle } from 'drizzle-orm/neon-serverless';
import { Pool, neon, neonConfig } from '@neondatabase/serverless';
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { eq, inArray, asc, desc } from 'drizzle-orm';

// Configure Neon HTTP client for Cloudflare Workers
neonConfig.fetch = (...args) => {
  return fetch(...args);
};

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


function getDb(databaseUrl) {
    const queryClient = postgres(databaseUrl); // This needs 'postgres' import still for `pg-core` in Drizzle
    return drizzle(queryClient, { schema: {
        articles: $articles,
        sources: $sources,
    }});
}
// --- End Drizzle DB Client Setup ---


// --- SIMULATED fetchArticleContentWithBrowser (THIS IS THE FUNCTION THAT WORKS WITH SIMULATED DATA) ---
async function fetchArticleContentWithBrowser(url, env, currentArticleId, currentRunId) {
    console.error(`[ContentFetcher] DEBUG: Article ID ${currentArticleId}, Run ID ${currentRunId}: SIMULATING Browser Rendering API call for ${url}`);
    await new Promise(resolve => setTimeout(resolve, 500));
    const simulatedContent = `SIMULATED CONTENT for ${url} - Retrieved at ${new Date().toISOString()}. Run ID: ${currentRunId}`;
    console.error(`[ContentFetcher] DEBUG: Article ID ${currentArticleId}, Run ID ${currentRunId}: Successfully SIMULATED fetching content.`);
    return simulatedContent;
}
// --- END NEW SIMULATED FUNCTION ---


export default {
    async queue(
        batch,
        env,
        ctx
    ) {
        const db = getDb(env.DATABASE_URL);

        for (const message of batch.messages) {
            const { articleId, runId: receivedRunId } = message.body;
            const currentRunId = receivedRunId || 'unknown-run';
            
            try {
                console.error(`[ContentFetcher] DEBUG: Processing article ID ${articleId}, Run ID ${currentRunId}`);

                if (!db.query || !db.query.articles) {
                    throw new Error(`[ContentFetcher] CRITICAL ERROR: db.query.articles is undefined before findFirst! Run ID: ${currentRunId}`);
                }

                const article = await db.query.articles.findFirst({
                    where: (articles, { eq: dbEq }) => dbEq(articles.id, articleId)
                });

                if (!article || !article.url) {
                    console.warn(`[ContentFetcher] WARN: Article ID ${articleId}, Run ID ${currentRunId}: Not found or missing URL. Acknowledging.`);
                    message.ack();
                    continue;
                }

                const fullContent = await fetchArticleContentWithBrowser(article.url, env, articleId, currentRunId);

                if (fullContent) { // This will now always be true due to simulation
                    console.error(`[ContentFetcher] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Content fetched (simulated). Attempting DB update.`);
                    
                    try {
                        const now = new Date();
                        const formattedTimestamp = formatTimestampForPgWithoutTimeZone(now);

                        await db.execute(sql`
                            UPDATE ${$articles}
                            SET 
                                content = ${fullContent},
                                content_fetched_at = ${formattedTimestamp},
                                processing_status = ${'Content_Fetched'}
                            WHERE id = ${articleId}
                        `);
                        console.error(`[ContentFetcher] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Raw SQL update for content successful.`);
                    } catch (rawSqlError) {
                        throw new Error(`[ContentFetcher] CRITICAL ERROR: Raw SQL update for content failed! ${rawSqlError?.message || String(rawSqlError)}. Article ID: ${articleId}, Run ID: ${currentRunId}`);
                    }

                    console.error(`[ContentFetcher] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: DB updated. Publishing to Gemini queue.`);
                    await env.ARTICLE_GEMINI_PROCESS_QUEUE.send({ articleId, runId: currentRunId });
                    message.ack();
                } else {
                    console.error(`[ContentFetcher] ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Failed to fetch content (simulated to succeed, this should not happen). Retrying.`);
                    message.retry();
                }
            } catch (error) {
                console.error(`[ContentFetcher] CRITICAL ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Unhandled exception in queue handler: ${error?.message || String(error)}`);
                throw error;
            }
        }
    },
};