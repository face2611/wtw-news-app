import getArticleAnalysisprompt, { articleAnalysisSchema } from '../prompts/articleAnalysisprompt'; // Verify path
import { $articles, and, eq, gte, isNull, sql } from '@meridian/database';
import { createGoogleGenerativeAI } from '@ai-sdk/google'; // Used for model object
import { Env } from '../index'; // Assuming Env is correctly exported from src/index.ts
import { generateObject } from 'ai'; // Main AI SDK function
import { getArticleWithBrowser, getArticleWithFetch } from '../lib/puppeteer'; // Verify path
import { getDb } from '../lib/utils'; // Verify path
import { WorkflowEntrypoint, WorkflowStep, WorkflowEvent, WorkflowStepConfig } from 'cloudflare:workers';
import { err, ok } from 'neverthrow';
import { ResultAsync } from 'neverthrow';
import { DomainRateLimiter } from '../lib/rateLimiter'; // Verify path

// Define Params if this workflow expects any specific payload when created
// If not, it can be 'unknown' or a more specific empty type like {}
type Params = unknown; // Or define if specific params are passed

const dbStepConfig: WorkflowStepConfig = {
  retries: { limit: 3, delay: '1 second', backoff: 'linear' },
  timeout: '10 seconds', // Increased for potentially slower DB operations in CF
};

const articleScrapeStepConfig: WorkflowStepConfig = {
  retries: { limit: 2, delay: '3 seconds', backoff: 'exponential' }, // Fewer retries for scraping
  timeout: '90 seconds', // Increased for potential browser rendering
};

const geminiStepConfig: WorkflowStepConfig = {
  retries: { limit: 2, delay: '5 seconds', backoff: 'exponential' },
  timeout: '2 minutes', // Generous timeout for LLM calls
};


// Main workflow class
export class ProcessArticles extends WorkflowEntrypoint<Env, Params> {
  async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
    console.log('[ProcessArticles] Workflow run instance started.');
    const env = this.env;
    const db = getDb(env.DATABASE_URL);
    const google = createGoogleGenerativeAI({ apiKey: env.GOOGLE_API_KEY, baseURL: env.GOOGLE_BASE_URL });

    async function getUnprocessedArticles(opts: { limit?: number }) {
      console.log('[ProcessArticles] Attempting to fetch unprocessed articles from DB.');
      const articles = await db
        .select({
          id: $articles.id,
          url: $articles.url,
          title: $articles.title,
          publishedAt: $articles.publishDate, // Ensure this matches your schema if it's 'publishDate'
          // text: $articles.content, // Fetch content here if not fetched later
        })
        .from($articles)
        .where(
          and(
            isNull($articles.processedAt),
            gte($articles.publishDate, new Date(new Date().getTime() - 48 * 60 * 60 * 1000)), // Last 48 hours
            isNull($articles.failReason) // Only process articles that haven't failed before (or have a retry strategy)
          )
        )
        .limit(opts.limit ?? 10) // <<<< DEBUGGING: Process small batches
        .orderBy(sql`RANDOM()`); // Process in random order to avoid getting stuck on same bad article
      console.log(`[ProcessArticles] Found ${articles.length} unprocessed articles for this batch.`);
      return articles;
    }

    // get articles to process
    const articlesToFetchContentFor = await step.do('get_articles_to_process', dbStepConfig, async () => getUnprocessedArticles({ limit: 1 })); // Batch size

    if (articlesToFetchContentFor.length === 0) {
      console.log('[ProcessArticles] No articles to process in this batch. Exiting.');
      return;
    }

    const rateLimiter = new DomainRateLimiter<{
      id: number; url: string; title: string | null; publishedAt: Date | null;
    }>({ maxConcurrent: 5, globalCooldownMs: 1000, domainCooldownMs: 3000 }); // Reduced concurrency

    const articlesWithContent: Array<{ id: number; title: string; text: string; publishedTime?: string; }> = [];

    const trickyDomains = ['reuters.com', 'nytimes.com', 'politico.com', 'science.org', 'alarabiya.net', 'reason.com', 'telegraph.co.uk', 'lawfaremedia.org', 'liberation.fr', 'france24.com']; // Added .org to lawfaremedia

    console.log(`[ProcessArticles] Starting content fetching for ${articlesToFetchContentFor.length} articles.`);
    const articleContentFetchResults = await rateLimiter.processBatch(articlesToFetchContentFor, step, async (article, domain) => {
      console.log(`[ProcessArticles] [RateLimiter] Fetching content for article ID: ${article.id}, URL: ${article.url}`);
      if (article.url.toLowerCase().endsWith('.pdf')) {
        console.log(`[ProcessArticles] Article ID: ${article.id} is a PDF. Skipping content fetch.`);
        return { id: article.id, success: false, error: 'pdf_skipped' };
      }

      const result = await step.do(
        `scrape_article_content_${article.id}`, articleScrapeStepConfig,
        async () => {
          let articleData: { title: string; text: string; publishedTime?: string } | undefined = undefined;
          const originalTitle = article.title || "Unknown Title"; // Use title from DB

          try {
            if (trickyDomains.some(td => article.url.includes(td))) { // Check if domain is in trickyDomains
              console.log(`[ProcessArticles] Article ID: ${article.id} is from a tricky domain (${domain}). Using getArticleWithBrowser.`);
              const articleResult = await getArticleWithBrowser(env, article.url); // Pass full env
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
                const articleResult = await getArticleWithBrowser(env, article.url); // Pass full env
                if (articleResult.isErr()) {
                  console.error(`[ProcessArticles] Browser fetch failed after light fetch failed for ID ${article.id} (${article.url}): ${articleResult.error.error}`);
                  return { id: article.id, success: false, error: `BrowserFallbackError: ${articleResult.error.error}` };
                }
                articleData = articleResult.value;
              }
            }
            // Ensure title from DB is preserved if fetch doesn't return one or returns a different one
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
    for (const result of articleContentFetchResults) {
      if (result && result.success && result.data && typeof result.data.text === 'string') { // Check if result and result.data and text are defined
        articlesWithContent.push({
          id: result.id,
          title: result.data.title, // Use title from fetched data (which includes original as fallback)
          text: result.data.text,
          publishedTime: result.data.publishedTime,
        });
      } else {
        console.log(`[ProcessArticles] Failed to get content for article ID: ${result?.id}. Error: ${result?.error || 'Unknown reason'}. Updating DB.`);
        await step.do(`update_db_failed_content_fetch_${result?.id}`, dbStepConfig, async () => {
          if (result?.id) { // Check if result.id is defined
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
      articlesWithContent.map(async article => { // Changed from articlesToProcess
        console.log(`[ProcessArticles] [LLM Batch] Analyzing article ID: ${article.id}. Title: ${article.title}`);
        let analysisResult: any = null;
        let llmError: string | null = null;
        try {
          analysisResult = await step.do(
            `analyze_article_gemini_${article.id}`, geminiStepConfig,
            async () => {
              // Ensure article.text exists and is not empty before sending to LLM
              if (!article.text || article.text.trim() === "") {
                  console.warn(`[ProcessArticles] [LLM Batch] Article ID: ${article.id} has empty text. Skipping Gemini call.`);
                  return { object: { completeness: 'PARTIAL_USELESS', relevance: 'NOISE', language: 'unknown', location: 'unknown', summary: { headline: 'No Content', entities: [], event: 'No Content', context: 'No Content' } } }; // Default object
              }
              const response = await generateObject({
                model: google('gemini-2.0-flash'), // <<<< CORRECTED MODEL NAME
                temperature: 0,
                prompt: getArticleAnalysisPrompt(article.title, article.text),
                schema: articleAnalysisSchema,
              });
              return response.object; // Assuming this is an object like { completeness, relevance, ... }
            }
          );
          console.log(`[ProcessArticles] [LLM Batch] Gemini analysis successful for article ID: ${article.id}`);
        } catch (e:any) {
            llmError = e?.message || String(e);
            console.error(`[ProcessArticles] [LLM Batch] Gemini analysis FAILED for article ID: ${article.id}:`, llmError);
        }

        console.log(`[ProcessArticles] [LLM Batch] Updating DB for article ID: ${article.id}`);
        await step.do(`update_db_after_llm_${article.id}`, dbStepConfig, async () => {
          if (analysisResult && !llmError) {
            await db
              .update($articles)
              .set({
                processedAt: new Date(),
                content: article.text, // Update content if it was fetched/modified
                title: article.title,   // Update title if it was fetched/modified
                completeness: analysisResult.completeness,
                relevance: analysisResult.relevance,
                language: analysisResult.language,
                location: analysisResult.location,
                summary: (() => {
                  if (analysisResult.summary === undefined || analysisResult.summary === null) return null;
                  let txt = '';
                  txt += `HEADLINE: ${analysisResult.summary.headline?.trim() || ''}\n`;
                  txt += `ENTITIES: ${(analysisResult.summary.entities || []).join(', ')}\n`;
                  txt += `EVENT: ${analysisResult.summary.event?.trim() || ''}\n`;
                  txt += `CONTEXT: ${analysisResult.summary.context?.trim() || ''}\n`;
                  return txt.trim() || null; // Return null if empty
                })(),
                failReason: null, // Clear previous errors
              })
              .where(eq($articles.id, article.id))
              .execute(); // Drizzle v0.3+ uses .execute()
             console.log(`[ProcessArticles] [LLM Batch] DB updated successfully for article ID: ${article.id}`);
          } else {
            await db
              .update($articles)
              .set({ processedAt: new Date(), failReason: `LLM Error: ${llmError?.substring(0,200) || 'Unknown LLM failure'}` })
              .where(eq($articles.id, article.id))
              .execute();
            console.log(`[ProcessArticles] [LLM Batch] DB updated with LLM failure for article ID: ${article.id}`);
          }
        });
      })
    );

    console.log(`[ProcessArticles] Finished LLM processing for ${articlesWithContent.length} articles.`);

    // Check if there are more articles to process still (for recursive self-triggering)
    console.log('[ProcessArticles] Checking for remaining unprocessed articles to potentially re-trigger workflow.');
    const remainingArticles = await step.do('get_remaining_articles_after_batch', dbStepConfig, async () =>
      getUnprocessedArticles({ limit: 10 }) // Check for another small batch
    );
    if (remainingArticles.length > 0) {
      console.log(`[ProcessArticles] Found at least ${remainingArticles.length} remaining articles to process. Re-triggering self.`);
      await step.do('retrigger_self_article_processor', dbStepConfig, async () => { // Changed step name
        // Use this.env to access the PROCESS_ARTICLES binding
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
    env.PROCESS_ARTICLES.create({ id: crypto.randomUUID() }), // Removed params if not used by ProcessArticles.run
    e => e instanceof Error ? e : new Error(String(e))
  );
  if (workflow.isErr()) {
    console.error('[Workflow Starter] Error creating PROCESS_ARTICLES workflow instance:', workflow.error);
    return err(workflow.error);
  }
  console.log(`[Workflow Starter] PROCESS_ARTICLES workflow instance created successfully with ID: ${workflow.value.id}`);
  return ok(workflow.value);
}