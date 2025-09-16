import app from './app';
// Explicitly import Workflow and Queue types for clarity if your TS setup doesn't auto-import/globals
import { Queue, Workflow, ScheduledController, ExecutionContext, ExportedHandler } from 'cloudflare:workers';

export type Env = {
  // Workflow Bindings for this project's workflows
  SCRAPE_RSS_FEED: typeof Workflow<any, any>;
  // PROCESS_ARTICLES: typeof Workflow<any, any>; // <<< REMOVED: This worker (wtw-production) no longer triggers ProcessArticles directly.
                                                 // Keep this commented out. If processArticles.workflow.ts exists,
                                                 // its export will still allow it to be deployed as a workflow.

  // Secrets needed by this worker (wtw-production)
  DATABASE_URL: string;
  MERIDIAN_SECRET_KEY: string;
  CORS_ORIGIN: string;

  // Queue Producer Binding for the first queue
  ARTICLE_CONTENT_FETCH_QUEUE: Queue;
};

export default {
  fetch: app.fetch,
  async scheduled({ cron }: ScheduledController, env: Env, ctx: ExecutionContext) {
    // - Every hour (at minute 4): trigger scraping of RSS feeds
    if (cron === '4 * * * *') {
      console.log('Scheduled cron: Initiating ScrapeRssFeed workflow creation...');
      // The .create method triggers the workflow's 'run' method asynchronously on the Cloudflare platform.
      // The 'env' object will be available to the workflow via 'this.env'.
      await env.SCRAPE_RSS_FEED.create({ id: crypto.randomUUID(), params: { force: true } });
      console.log('ScrapeRssFeed workflow instance successfully created. Check worker logs for its execution.');
      // The lines `await runScrapeRssFeedWorkflow(env, ctx);` and `console.log("RSS feed scraping finished.");`
      // were removed here as they are redundant or syntactically incorrect.
    }
    // The 'return;' should be outside the 'if' block if it's meant to end the scheduled handler.
    // If there were other cron patterns, they would go below this 'if' block.
    return;
  },
} satisfies ExportedHandler<Env>;

// These exports ensure that Cloudflare recognizes `ScrapeRssFeed` and `ProcessArticles`
// as deployable workflows that are part of this worker project.
// Even if `wtw-production` no longer directly uses `ProcessArticles`, it still needs
// to be exported here if it's defined as a WorkflowEntrypoint in `./workflows/processArticles.workflow.ts`
// and you want it to be deployable.
export { ScrapeRssFeed } from './workflows/rssFeed.workflow';
export { ProcessArticles } from './workflows/processArticles.workflow'; // Keep this line.