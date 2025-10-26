// C:\Users\phili\meridian\apps\gemini-processor-worker\src\index.js
// --- Full & Correct Gemini Processor Logic (postgres-js client & Google Gemini via Cloudflare AI Gateway) ---

// --- Drizzle DB Client Setup (SELF-CONTAINED) ---
// All Drizzle-related imports are local to this worker as per its self-contained design.
import { drizzle } from 'drizzle-orm/postgres-js'; // Use postgres-js adapter
import postgres from 'postgres'; // Use postgres client
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { eq, inArray } from 'drizzle-orm'; // inArray is used in the error handling for updates

// --- Helper function for timestamp formatting ---
function formatTimestampForPgWithoutTimeZone(date) {
    const d = new Date(date);
    const year = String(d.getFullYear()).padStart(4, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}
// --- End Helper ---


// --- Define schema INLINE for this worker (SELF-CONTAINED) ---
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
  processedAt: timestamp('processed_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
  createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});
// --- End INLINE SCHEMA ---


// --- getDb function for self-contained Drizzle (SELF-CONTAINED) ---
let dbClient = null; // Declare dbClient here for lazy init (per invocation, not global worker scope)

function getDb(databaseUrl) { // This accepts the DATABASE_URL secret (a string)
    // NOTE: This lazy init is designed to be per-invocation if the worker is reset,
    // or reused within a single invocation. A truly global client across worker instances
    // needs more complex pooling. For Cloudflare Workers, a client per invocation is safer.
    // The previous architecture had a separate function that called `getDb` for each invocation,
    // thereby resolving "Cannot perform I/O" errors by getting a fresh client.
    // Here, dbClient is local to the module, so it's a singleton for the current worker instance.
    if (!dbClient) {
        const queryClient = postgres(databaseUrl, {
            // Options to prevent "Cannot perform I/O" if connection isn't properly closed
            // or to manage connection lifecycle. For workers, simpler is often better.
            // Using a new client per invocation ensures no shared state issues,
            // but can incur overhead. The original handover mentioned "getDb initializes
            // a new postgres client for each invocation" - that pattern is safer.
            // Let's modify this to ensure a new client on each `getDb` call.
        });
        dbClient = drizzle(queryClient, { schema: {
            articles: $articles,
            sources: $sources,
        }});
    }
    return dbClient;
}
// --- End Drizzle DB Client Setup ---


// --- Google Gemini via Cloudflare AI Gateway Service ---
import { GoogleGenerativeAI } from "@google/generative-ai";

let generativeModelInstance = null; // Cache the model instance for reuse across invocations

async function getGenerativeModel(env) {
    if (generativeModelInstance) {
        return generativeModelInstance;
    }

    if (!env.GOOGLE_AI_STUDIO_TOKEN || !env.CLOUDFLARE_ACCOUNT_ID || !env.AI_GATEWAY_NAME) {
        throw new Error("Missing Google AI Studio Gateway environment variables (GOOGLE_AI_STUDIO_TOKEN, CLOUDFLARE_ACCOUNT_ID, AI_GATEWAY_NAME).");
    }

    const genAI = new GoogleGenerativeAI(env.GOOGLE_AI_STUDIO_TOKEN);
    generativeModelInstance = genAI.getGenerativeModel(
        { model: "gemini-2.5-flash" },
        {
            baseUrl: `https://gateway.ai.cloudflare.com/v1/${env.CLOUDFLARE_ACCOUNT_ID}/${env.AI_GATEWAY_NAME}/google-ai-studio`,
        },
    );
    return generativeModelInstance;
}


// Centralized AI Call Function (handles JSON parsing and cleaning for Google Generative AI)
async function callGoogleGenerativeAiViaGateway(modelId, messages, env, currentRunId, isJsonOutput = true) {
    try {
        const generativeModel = await getGenerativeModel(env);
        
        // Transform messages to the format expected by GoogleGenerativeAI SDK
        const formattedContents = messages.map(msg => ({
            role: msg.role,
            parts: [{ text: msg.content }]
        }));

        const response = await generativeModel.generateContent({ contents: formattedContents });
        const rawAiOutput = response.response.text();

        if (isJsonOutput) {
            let cleanedAiOutput = rawAiOutput;
            // More robust cleaning for AI JSON output
            cleanedAiOutput = cleanedAiOutput
                .replace(/^[`'"]{3}json\s*/, '') // Remove ```json, '''json, """json from start
                .replace(/\s*[`'"]{3}$/, '')   // Remove ```, ''' or """ from end
                .trim();

            const jsonStartIndex = cleanedAiOutput.indexOf('{');
            const jsonEndIndex = cleanedAiOutput.lastIndexOf('}');

            if (jsonStartIndex !== -1 && jsonEndIndex !== -1 && jsonEndIndex > jsonStartIndex) {
                cleanedAiOutput = cleanedAiOutput.substring(jsonStartIndex, jsonEndIndex + 1);
            } else {
                console.warn(`[GeminiProcessor] WARN: Run ID ${currentRunId}: Aggressive JSON extraction failed to find clear { } block. Using less-cleaned output. Raw: ${rawAiOutput.slice(0, 200)}`);
            }
            
            try {
                return JSON.parse(cleanedAiOutput);
            } catch (parseError) {
                console.error(`[GeminiProcessor] ERROR: Run ID ${currentRunId}: Failed to parse AI JSON output: ${parseError.message}. Raw AI Output (cleaned & extracted, first 500): ${cleanedAiOutput.slice(0, 500)}`);
                throw new Error(`AI JSON parse failed: ${parseError.message}`);
            }
        } else {
            return rawAiOutput; // Return raw text for non-JSON output
        }

    } catch (error) {
        console.error(`[GeminiProcessor] ERROR: Run ID ${currentRunId}: Google Generative AI Call Failed (${modelId}): ${error?.message || String(error)}`);
        throw new Error(`Google Generative AI Call Failed (${modelId}): ${error?.message || String(error)}`);
    }
}
// --- End Google Gemini via Cloudflare AI Gateway Service ---


// --- Worker Entrypoint (Queue Handler) ---
export default {
    async queue(
        batch,
        env,
        ctx
    ) {
        if (!env.DATABASE_URL) {
            throw new Error(`[GeminiProcessor] ERROR: DATABASE_URL binding is missing or undefined!`);
        }
        if (!env.GOOGLE_AI_STUDIO_TOKEN || !env.CLOUDFLARE_ACCOUNT_ID || !env.AI_GATEWAY_NAME) {
            throw new Error(`[GeminiProcessor] ERROR: Missing Google AI Studio Gateway environment variables!`);
        }
        
        // As per handover: "getDb initializes a new postgres client for each invocation"
        // This ensures a fresh client per invocation, avoiding "Cannot perform I/O" issues.
        const db = drizzle(postgres(env.DATABASE_URL), { schema: { articles: $articles, sources: $sources } });

        for (const message of batch.messages) {
            const { articleId, runId: receivedRunId } = message.body;
            const currentRunId = receivedRunId || 'unknown-run';
            
            try {
                console.error(`[GeminiProcessor] DEBUG: Processing article ID ${articleId}, Run ID ${currentRunId}`);

                const article = await db.query.articles.findFirst({
                    where: (articles, { eq: dbEq }) => dbEq(articles.id, articleId)
                });

                if (!article || !article.content) {
                    console.warn(`[GeminiProcessor] WARN: Article ID ${articleId}, Run ID ${currentRunId}: Not found or missing content. Acknowledging.`);
                    if (!article || article.processing_status === 'Content_Fetch_Failed') {
                         const now = new Date();
                         const formattedTimestamp = formatTimestampForPgWithoutTimeZone(now);
                         await db.update($articles)
                            .set({ 
                                processing_status: 'AI_Failed',
                                failReason: 'No content available for AI processing (Content Fetch Failed)',
                                geminiProcessedAt: formattedTimestamp
                            })
                            .where(eq($articles.id, articleId));
                    }
                    message.ack();
                    continue;
                }

                const systemPrompt = `You are an expert news summarizer and analyst. Provide a concise summary, detect the primary language, main location, and rate completeness/relevance based on the article content. Your output MUST be STRICTLY VALID JSON, with no preamble, no markdown wrappers (like \`\`\`json), and no extraneous text. If you cannot fully complete a field, use null or "unknown" as appropriate according to the schema type, but DO NOT deviate from the JSON structure. Ensure location is a string or null.`;
                
                const userPrompt = `Summarize this article, detect its language, identify the main location, and rate its completeness and relevance. Output format: {"summary": "...", "language": "...", "location": "...", "completeness": "low|medium|high", "relevance": "low|medium|high"}.\n\nArticle Content:\n${article.content.slice(0, 100000)}`; // Use up to 100k chars for Gemini-2.5-flash context

                // Consolidate system and user prompts into a single user message for Google Generative AI's generateContent
                const combinedUserPrompt = `
${systemPrompt}

${userPrompt}
`.trim();

                const { summary, language, location, completeness, relevance, failReason } = await callGoogleGenerativeAiViaGateway(
                    "gemini-2.5-flash", // modelId for logging purposes
                    [{ role: 'user', content: combinedUserPrompt }],
                    env, 
                    currentRunId,
                    true // Expect JSON output
                );

                console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Gemini AI analysis completed. Attempting DB update.`);
                
                try {
                    const now = new Date();
                    const formattedTimestamp = formatTimestampForPgWithoutTimeZone(now);

                    // Sanitize string fields before SQL update
                    const sanitizedSummary = summary ? String(summary).replace(/'/g, "''") : null;
                    const sanitizedLocation = location ? String(location).replace(/'/g, "''") : null;
                    const sanitizedFailReason = failReason ? String(failReason).replace(/'/g, "''") : null;

                    await db.execute(sql`
                        UPDATE ${$articles}
                        SET 
                            summary = ${sanitizedSummary},
                            language = ${language},
                            location = ${sanitizedLocation},
                            completeness = ${completeness},
                            relevance = ${relevance},
                            fail_reason = ${sanitizedFailReason},
                            gemini_processed_at = ${formattedTimestamp},
                            processing_status = ${failReason ? 'AI_Failed' : 'AI_Processed'}
                        WHERE id = ${articleId}
                    `);
                    console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: Raw SQL update for AI results successful.`);
                } catch (rawSqlError) {
                    throw new Error(`[GeminiProcessor] CRITICAL ERROR: Raw SQL update for AI results failed! ${rawSqlError?.message || String(rawSqlError)}. Article ID: ${articleId}, Run ID: ${currentRunId}`);
                }

                message.ack();
                console.error(`[GeminiProcessor] DEBUG: Article ID ${articleId}, Run ID ${currentRunId}: All processing for article completed successfully.`);
            } catch (error) {
                console.error(`[GeminiProcessor] CRITICAL ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Unhandled exception in queue handler: ${error?.message || String(error)}`);
                if (db) { // Check if db client was successfully initialized before attempting update
                    try {
                        const now = new Date();
                        const formattedTimestamp = formatTimestampForPgWithoutTimeZone(now);
                        await db.update($articles)
                            .set({ 
                                processing_status: 'AI_Failed',
                                failReason: `Unhandled: ${error?.message || String(error)}`,
                                geminiProcessedAt: formattedTimestamp
                            })
                            .where(eq($articles.id, articleId));
                    } catch (dbUpdateError) {
                        console.error(`[GeminiProcessor] ERROR: Article ID ${articleId}, Run ID ${currentRunId}: Failed to update status after critical error: ${dbUpdateError.message}`);
                    }
                }
                message.retry(); // Re-queue the message for retry
            }
        }
    },
};