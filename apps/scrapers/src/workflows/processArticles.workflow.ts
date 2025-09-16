import getArticleAnalysisPrompt, { articleAnalysisSchema } from '../prompts/articleAnalysisPrompt'; // Verify path
import { $articles, and, eq, gte, isNull, sql } from '@meridian/database';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { Env } from '../index'; // Assuming Env is correctly exported
import { generateObject } from 'ai';
import { getArticleWithBrowser, getArticleWithFetch } from '../lib/puppeteer'; // Verify path
import { getDb } from '../lib/utils'; // Verify path
import { WorkflowEntrypoint, WorkflowStep, WorkflowEvent, WorkflowStepConfig } from 'cloudflare:workers';
import { err, ok } from 'neverthrow';
import { ResultAsync } from 'neverthrow';
import { DomainRateLimiter } from '../lib/rateLimiter'; // Verify path

// Define Params if this workflow expects any specific payload when created
type Params = unknown; // Or define if specific params are passed

// --- Define Step Configurations ---
const dbStepConfig: WorkflowStepConfig = {
    retries: { limit: 3, delay: '1 second', backoff: 'linear' },
    timeout: '20 seconds', // Increased slightly for DB operations
};

const articleScrapeStepConfig: WorkflowStepConfig = {
    retries: { limit: 2, delay: '5 seconds', backoff: 'exponential' }, // Fewer retries for scraping
    timeout: '120 seconds', // 2 minutes for content fetching (especially browser)
};

const geminiStepConfig: WorkflowStepConfig = {
    retries: { limit: 2, delay: '10 seconds', backoff: 'exponential' }, // Retries with longer delay for API limits
    timeout: '180 seconds', // 3 minutes for Gemini calls, generous
};


// Main workflow class
export class ProcessArticles extends WorkflowEntrypoint<Env, Params> {
    async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
        console.log('[ProcessArticles] Workflow run instance started.'); // LOG START
        const env = this.env;
        const db = getDb(env.DATABASE_URL);
        // Initialize google client once if it's stateless, or per call if stateful/config changes
        // Assuming createGoogleGenerativeAI is cheap to call or the object is lightweight
        const google = createGoogleGenerativeAI({ apiKey: env.GOOGLE_API_KEY, baseURL: env.GOOGLE_BASE_URL });

        async function getUnprocessedArticles(opts: { limit?: number }) {
            console.log('[ProcessArticles] Attempting to fetch unprocessed articles from DB.');
            const articles = await db
                .select({
                    id: $articles.id,
                    url: $articles.url,
                    title: $articles.title,
                    publishedAt: $articles.publishDate,
                })
                .from($articles)
                .where(
                    and(
                        isNull($articles.processedAt),
                        //gte($articles.publishDate, new Date(new Date().getTime() - 48 * 60 * 60 * 1000)),
                        eq($articles.failReason, 'Too many subrequests'), 
                        eq($articles.failReason, 'Error: Too many subrequests') 

                        //isNull($articles.failReason) // Or specific retry logic for certain failReasons
                    )
                )
                .limit(opts.limit ?? 2) // <<<< DEBUGGING: Start with a very small batch (e.g., 1 or 2)
                .orderBy(sql`RANDOM()`);
            console.log(`[ProcessArticles] Found ${articles.length} unprocessed articles for this batch.`);
            return articles;
        }

        // Get articles to process
        // Use the defined dbStepConfig for this database call
        const articlesToFetchContentFor = await step.do('get_articles_to_process', dbStepConfig, async () => getUnprocessedArticles({ limit: 2 })); // <<<< Make sure limit is small for testing

        if (articlesToFetchContentFor.length === 0) {
            console.log('[ProcessArticles] No articles to process in this batch. Exiting.');
            return;
        }

        // Create rate limiter
        const rateLimiter = new DomainRateLimiter<{
            id: number; url: string; title: string | null; publishedAt: Date | null;
        }>({ maxConcurrent: 3, globalCooldownMs: 1000, domainCooldownMs: 5000 }); // Reduced concurrency

        const articlesWithContent: Array<{ id: number; title: string; text: string; publishedTime?: string; }> = [];

        const trickyDomains = ['reuters.com', /* ... other domains ... */ 'france24.com'];

        console.log(`[ProcessArticles] Starting content fetching for ${articlesToFetchContentFor.length} articles.`);
        const articleContentFetchResults = await rateLimiter.processBatch(articlesToFetchContentFor, step, async (article, domain) => {
            console.log(`[ProcessArticles] [RateLimiter] Fetching content for article ID: ${article.id}, URL: ${article.url}`);
            if (article.url.toLowerCase().endsWith('.pdf')) {
                console.log(`[ProcessArticles] Article ID: ${article.id} is a PDF. Skipping content fetch.`);
                return { id: article.id, success: false, error: 'pdf_skipped' };
            }

            // Use the articleScrapeStepConfig for this potentially long step
            const result = await step.do(
                `scrape_article_content_${article.id}`,
                articleScrapeStepConfig, // <<<< APPLYING DEFINED CONFIG
                async () => {
                    // ... (rest of your existing scrape logic for a single article) ...
                    let articleData: { title: string; text: string; publishedTime: string | undefined } | undefined = undefined;
                    const originalTitle = article.title || "Unknown Title";

                    try {
                        console.log(`[ProcessArticles] Article ID ${article.id}: Content fetch logic initiated.`);
                        if (trickyDomains.some(td => article.url.includes(td))) {
                            console.log(`[ProcessArticles] Article ID: ${article.id} is from a tricky domain (${domain}). Using getArticleWithBrowser.`);
                            const articleResult = await getArticleWithBrowser(env, article.url);
                            if (articleResult.isErr()) {
                                console.error(`[ProcessArticles] Browser fetch failed for ID ${article.id} (${article.url}): ${articleResult.error.error}`);
                                return { id: article.id, success: false, error: `BrowserFetchError: ${articleResult.error.error}` };
                            }
                            articleData = articleResult.value;
                        } else {
                            console.log(`[ProcessArticles] Article ID: ${article.id} attempting light fetch for ${article.url}`);
                            const lightResult = await getArticleWithFetch(article.url);
                            if (lightResult.isOk()) {
                                articleData = lightResult.value;
                                console.log(`[ProcessArticles] Light fetch successful for ID ${article.id}`);
                            } else {
                                console.warn(`[ProcessArticles] Light fetch failed for ID ${article.id} (${article.url}): ${lightResult.error.message}. Trying browser.`);
                                const jitterTime = Math.random() * 2500 + 500;
                                await step.sleep(`jitter_before_browser_${article.id}`, jitterTime);
                                const articleResult = await getArticleWithBrowser(env, article.url);
                                if (articleResult.isErr()) {
                                    console.error(`[ProcessArticles] Browser fetch failed after light fetch failed for ID ${article.id} (${article.url}): ${articleResult.error.error}`);
                                    return { id: article.id, success: false, error: `BrowserFallbackError: ${articleResult.error.error}` };
                                }
                                articleData = articleResult.value;
                            }
                        }
                        console.log(`[ProcessArticles] Article ID ${article.id}: Content fetch successful.`);
                        return { id: article.id, success: true, data: { ...articleData, title: articleData?.title || originalTitle, text: articleData?.text || "" } };
                    } catch (e: any) {
                        console.error(`[ProcessArticles] Unhandled exception during content fetch for article ID ${article.id}: ${e.message}`);
                        return { id: article.id, success: false, error: `UnhandledFetchException: ${e.message}` };
                    }
                }
            );
            return result;
        });

        console.log('[ProcessArticles] Content fetching batch complete.');
        // ... (rest of your existing logic to handle results and populate articlesWithContent) ...
        for (const result of articleContentFetchResults) {
            if (result && result.success && result.data && typeof result.data.text === 'string') {
                articlesWithContent.push({
                    id: result.id,
                    title: result.data.title,
                    text: result.data.text,
                    publishedTime: result.data.publishedTime,
                });
            } else {
                console.log(`[ProcessArticles] Failed to get content for article ID: ${result?.id}. Error: ${result?.error || 'Unknown reason'}. Updating DB.`);
                await step.do(`update_db_failed_content_fetch_${result?.id}`, dbStepConfig, async () => {
                    if (result?.id) {
                        await db
                            .update($articles)
                            .set({ processedAt: new Date(), failReason: String(result.error ? result.error : 'Content fetch failed') })
                            .where(eq($articles.id, result.id));
                    }
                });
            }
        }

        if (articlesWithContent.length === 0) {
            console.log('[ProcessArticles] No articles with content to process with LLM after fetching. Exiting.');
            return;
        }
        console.log(`[ProcessArticles] ${articlesWithContent.length} articles have content and are ready for LLM processing.`);


        // process with LLM
        console.log('[ProcessArticles] Starting LLM processing for collected articles.');
        await Promise.all(
            articlesWithContent.map(async article => {
                console.log(`[ProcessArticles] [LLM Batch] Analyzing article ID: ${article.id}. Title: ${article.title}`);
                let analysisResultObject: any = null; // Renamed from articleAnalysis to avoid conflict
                let llmError: string | null = null;
                try {
                    // Use the geminiStepConfig for this potentially long step
                    analysisResultObject = await step.do(
                        `analyze_article_gemini_${article.id}`,
                        geminiStepConfig, // <<<< APPLYING DEFINED CONFIG
                        async () => {
                            if (!article.text || article.text.trim() === "") {
                                console.warn(`[ProcessArticles] [LLM Batch] Article ID: ${article.id} has empty text. Skipping Gemini call.`);
                                // Return a default structure that matches articleAnalysisSchema
                                return { object: { completeness: 'PARTIAL_USELESS', relevance: 'NOISE', language: 'unknown', location: 'unknown', summary: { headline: 'No Content', entities: [], event: 'No Content', context: 'No Content' } } };
                            }
                            console.log(`[ProcessArticles] [LLM Batch] Calling Gemini for article ID: ${article.id}. Prompt length: ${getArticleAnalysisPrompt(article.title, article.text).length}`);
                            const response = await generateObject({
                                model: google('gemini-1.5-flash'), // <<<< YOUR MODEL NAME
                                temperature: 0,
                                prompt: getArticleAnalysisPrompt(article.title, article.text),
                                schema: articleAnalysisSchema,
                            });
                            return response.object;
                        }
                    );
                    console.log(`[ProcessArticles] [LLM Batch] Gemini analysis successful for article ID: ${article.id}`);
                } catch (e: any) {
                    llmError = e?.message || String(e); // Get error message
                    console.error(`[ProcessArticles] [LLM Batch] Gemini analysis FAILED for article ID: ${article.id}:`, llmError);
                    // Log the full error object if it's more complex
                    if (typeof e === 'object' && e !== null) console.error("Full Gemini error object:", JSON.stringify(e, Object.getOwnPropertyNames(e)));

                }

                console.log(`[ProcessArticles] [LLM Batch] Updating DB for article ID: ${article.id}`);
                // Use dbStepConfig for database updates
                await step.do(`update_db_after_llm_${article.id}`, dbStepConfig, async () => {
                    if (analysisResultObject && !llmError) { // Check analysisResultObject
                        await db
                            .update($articles)
                            .set({
                                processedAt: new Date(),
                                content: article.text,
                                title: article.title,
                                completeness: analysisResultObject.completeness,
                                relevance: analysisResultObject.relevance,
                                language: analysisResultObject.language,
                                location: analysisResultObject.location,
                                summary: (() => {
                                    if (analysisResultObject.summary === undefined || analysisResultObject.summary === null) return null;
                                    let txt = '';
                                    txt += `HEADLINE: ${analysisResultObject.summary.headline?.trim() || ''}\n`;
                                    txt += `ENTITIES: ${(analysisResultObject.summary.entities || []).join(', ')}\n`;
                                    txt += `EVENT: ${analysisResultObject.summary.event?.trim() || ''}\n`;
                                    txt += `CONTEXT: ${analysisResultObject.summary.context?.trim() || ''}\n`;
                                    return txt.trim() || null;
                                })(),
                                failReason: null,
                            })
                            .where(eq($articles.id, article.id))
                            .execute(); // Add .execute() if Drizzle v0.300.0+
                        console.log(`[ProcessArticles] [LLM Batch] DB updated successfully for article ID: ${article.id}`);
                    } else {
                        await db
                            .update($articles)
                            .set({ processedAt: new Date(), failReason: `LLM Error: ${llmError?.substring(0, 250) || 'Unknown LLM failure'}` }) // Ensure substring doesn't exceed column length
                            .where(eq($articles.id, article.id))
                            .execute(); // Add .execute()
                        console.log(`[ProcessArticles] [LLM Batch] DB updated with LLM failure for article ID: ${article.id}`);
                    }
                });
            })
        );

        console.log(`[ProcessArticles] Finished LLM processing for ${articlesWithContent.length} articles.`);

        // Check for more articles to process & self-retrigger
        console.log('[ProcessArticles] Checking for remaining unprocessed articles to potentially re-trigger workflow.');
        const remainingArticles = await step.do('get_remaining_articles_after_batch', dbStepConfig, async () =>
            getUnprocessedArticles({ limit: 2 }) // <<<< DEBUGGING: Check for a small number to re-trigger
        );
        if (remainingArticles.length > 0) {
            console.log(`[ProcessArticles] Found at least ${remainingArticles.length} remaining articles to process. Re-triggering self.`);
            await step.do('retrigger_self_article_processor', dbStepConfig, async () => {
                const workflow = await this.env.PROCESS_ARTICLES.create({ id: crypto.randomUUID() });
                console.log(`[ProcessArticles] Self re-triggered. New workflow ID: ${workflow.id}`);
                return workflow.id;
            });
        } else {
            console.log('[ProcessArticles] No more remaining articles to process in this cycle.');
        }
        console.log('[ProcessArticles] Workflow run instance finished.');
    }
}

// helper to start the workflow from elsewhere
export async function startProcessArticleWorkflow(env: Env) {
    console.log('[Workflow Starter] Attempting to start ProcessArticles workflow.');
    const workflow = await ResultAsync.fromPromise(
        env.PROCESS_ARTICLES.create({ id: crypto.randomUUID() }),
        e => e instanceof Error ? e : new Error(String(e))
    );
    if (workflow.isErr()) {
        console.error('[Workflow Starter] Error creating PROCESS_ARTICLES workflow instance:', workflow.error);
        return err(workflow.error);
    }
    console.log(`[Workflow Starter] PROCESS_ARTICLES workflow instance created successfully with ID: ${workflow.value.id}`);
    return ok(workflow.value);
}