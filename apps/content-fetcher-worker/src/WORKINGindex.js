// C:\Users\phili\meridian\apps\content-fetcher-worker\src\index.js

// --- Drizzle DB Client Setup (SELF-CONTAINED FOR DEBUGGING) ---
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
// REMOVED: import * as importedSchema from '../../../packages/database/src/schema'; // <<<< REMOVED - Using inline schema
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core'; // <<<< Import Drizzle types for INLINE SCHEMA
import { sql } from 'drizzle-orm'; // <<<< CORRECTED IMPORT FOR SQL
import { eq, inArray } from 'drizzle-orm'; // Keep individual Drizzle ops

// --- NEW: Define schema INLINE for this worker for maximum isolation ---
// This is a direct copy of your relevant schema
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
  content: text('content'), // This is the problematic column
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
// --- END NEW INLINE SCHEMA ---


function getDb(databaseUrl) {
    const queryClient = postgres(databaseUrl);
    // <<<< NOW USE THE INLINE SCHEMA DEFINED ABOVE ($articles, $sources)
    return drizzle(queryClient, { schema: {
        articles: $articles, // Use the directly defined $articles
        sources: $sources,   // Use the directly defined $sources
    }});
}
// --- End Drizzle DB Client Setup ---


// --- NEW: SIMULATED fetchArticleContentWithBrowser (no change needed here) ---
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

                // The check for db.query.articles is still useful
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
                    
                    // --- NEW: Separate update for 'content' using raw SQL ---
                    try {
                        // Update 'content', 'content_fetched_at', and 'processing_status' using raw SQL
                        await db.execute(sql`
                            UPDATE ${$articles}
                            SET 
                                content = ${fullContent}::text,
                                content_fetched_at = ${new Date().toISOString()}::timestamp without time zone,
                                processing_status = ${'Content_Fetched'}::text
                            WHERE id = ${articleId}
                        `);
                        console.error(`[ContentFetcher] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Raw SQL update for content successful.`);
                    } catch (rawSqlError) {
                        throw new Error(`[ContentFetcher] CRITICAL ERROR: Raw SQL update for content failed! ${rawSqlError?.message || String(rawSqlError)}. Article ID: ${articleId}, Run ID: ${currentRunId}`);
                    }
                    // --- END NEW RAW SQL UPDATE ---

                    console.error(`[ContentFetcher] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: DB updated (partially with raw SQL). Publishing to Gemini queue.`);
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