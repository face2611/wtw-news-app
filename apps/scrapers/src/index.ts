// C:\Users\phili\meridian\apps\scrapers\src\index.ts

import app from './app';
import { Queue, ScheduledController, ExecutionContext, ExportedHandler } from 'cloudflare:workers';
// Import the new function directly
import { runScrapeRssFeedLogic } from './logic/rssFeed.logic'; // <<<< CHANGED IMPORT

export type Env = {
  
  // ... (PROCESS_ARTICLES should still be commented out) ...

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
    if (cron === '4 * * * *') {
      console.log('Scheduled cron: Directly running ScrapeRssFeed logic...'); // <<<< LOG CHANGE
      // Directly call the new function
      await runScrapeRssFeedLogic(env, ctx, { force: true }); // <<<< CHANGED CALL, pass env, ctx, and params
      console.log('Scheduled cron: ScrapeRssFeed logic finished.'); // <<<< LOG CHANGE
    }
    return;
  },
} satisfies ExportedHandler<Env>;


// Keep this if processArticles.workflow.ts exists and you want it deployable (even if not triggered by this worker)
export { ProcessArticles } from './workflows/processArticles.workflow';