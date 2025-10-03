// C:\Users\phili\meridian\apps\gemini-processor-worker\src\index.js

// --- Drizzle DB Client Setup (SELF-CONTAINED FOR DEBUGGING) ---
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm'; // Import sql for default values and raw queries
import { eq, inArray } from 'drizzle-orm'; // For queries

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

// --- NEW: Define schema INLINE for this worker for maximum isolation ---
// This is a direct copy of your relevant schema from packages/database/src/schema.ts
// These definitions are needed for Drizzle's query builder (e.g., findFirst, and in the raw SQL for table name)
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
// --- END NEW INLINE SCHEMA ---


function getDb(databaseUrl) {
    const queryClient = postgres(databaseUrl);
    return drizzle(queryClient, { schema: {
        articles: $articles,
        sources: $sources,
    }});
}
// --- End Drizzle DB Client Setup ---


// --- Google Gemini API Service (Simulated) ---
async function callGeminiApi(articleContent, env, currentArticleId, currentRunId) {
    console.error(`[GeminiProcessor] DEBUG: Article ID ${currentArticleId}, Run ID ${currentRunId}: SIMULATING Gemini API call for content length ${articleContent.length}`);
    try {
        await new Promise(resolve => setTimeout(resolve, 1000));

        const simulatedSummary = `Simulated summary for article with ID ${currentArticleId} and content starting: ${articleContent.substring(0, Math.min(articleContent.length, 100))}. Run ID: ${currentRunId}`;
        const simulatedRelevance = 'High';
        const simulatedLocations = ['Simulated City A', 'Simulated Country B']; // Ensure array for JSONB
        const simulatedCompleteness = 'Full';

        const simulatedResult = {
            summary: simulatedSummary,
            relevance: simulatedRelevance,
            locations: simulatedLocations,
            completeness: simulatedCompleteness,
        };

        console.error(`[GeminiProcessor] DEBUG: Article ID ${currentArticleId}, Run ID ${currentRunId}: Successfully SIMULATED Gemini analysis.`);
        return simulatedResult;

    } catch (error) {
        throw new Error(`Gemini API Exception: ${error?.message || String(error)}. Article ID: ${currentArticleId}, Run ID: ${currentRunId}`);
    }
}


// --- Worker Entrypoint (Queue Handler) ---
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
                console.error(`[GeminiProcessor] DEBUG: Processing article ID ${articleId}, Run ID ${currentRunId}`);

                // The check for db.query.articles is still useful
                if (!db.query || !db.query.articles) {
                    throw new Error(`[GeminiProcessor] CRITICAL ERROR: db.query.articles is undefined before findFirst! Run ID: ${currentRunId}`);
                }

                const article = await db.query.articles.findFirst({
                    where: (articles, { eq: dbEq }) => dbEq(articles.id, articleId)
                });

                if (!article || !article.content) {
                    console.warn(`[GeminiProcessor] WARN: Article ID ${articleId}, Run ID ${currentRunId}: Not found or missing content. Acknowledging.`);
                    message.ack();
                    continue;
                }

                // Call Gemini API for analysis (simulated for now)
                const geminiResult = await callGeminiApi(article.content, env, articleId, currentRunId);

                if (geminiResult) {
                    console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Gemini analysis (simulated) complete. Attempting DB update.`);
                    
                    // --- NEW: ULTIMATE RAW SQL UPDATE BYPASS (CORRECTED SYNTAX) ---
                    try {
                        const now = new Date();
                        const formattedTimestamp = formatTimestampForPgWithoutTimeZone(now);

                        const locationsToSave = Array.isArray(geminiResult.locations) ? geminiResult.locations : [];
                        
                        await db.execute(sql`
                            UPDATE ${$articles}
                            SET 
                                summary = ${geminiResult.summary},
                                relevance = ${geminiResult.relevance},
                                location = ${JSON.stringify(locationsToSave)},
                                completeness = ${geminiResult.completeness},
                                gemini_processed_at = ${formattedTimestamp},
                                processing_status = ${'Gemini_Processed'}
                            WHERE id = ${articleId}
                        `);
                        console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: ULTIMATE Raw SQL update for Gemini data successful.`);
                    } catch (rawSqlError) {
                        throw new Error(`[GeminiProcessor] CRITICAL ERROR: ULTIMATE Raw SQL update for Gemini data failed! ${rawSqlError?.message || String(rawSqlError)}. Article ID: ${articleId}, Run ID: ${currentRunId}`);
                    }
                    // --- END NEW ULTIMATE RAW SQL UPDATE ---

                    message.ack();
                    console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: All processing for article completed successfully.`);
                } else {
                    console.error(`[GeminiProcessor] ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Gemini analysis failed (simulated to succeed, this should not happen). Retrying.`);
                    message.retry();
                }
            } catch (error) {
                console.error(`[GeminiProcessor] CRITICAL ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Unhandled exception in queue handler: ${error?.message || String(error)}`);
                throw error;
            }
        }
    },
};