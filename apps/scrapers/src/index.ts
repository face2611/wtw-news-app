// C:\Users\phili\meridian\apps\scrapers\src\index.ts
import app from './app';
import { Queue, ScheduledController, ExecutionContext, ExportedHandler, Workflow } from 'cloudflare:workers'; // Ensure Workflow is imported
import { runScrapeRssFeedLogic } from './logic/rssFeed.logic'; // Assuming this is the direct function

export type Env = {
  // Drizzle/Hyperdrive expects the binding to be of type any, as its internal structure is complex
  DATABASE_HYPERDRIVE: any; // <<<< CRITICAL: Add Hyperdrive binding here
  // REMOVED: DATABASE_URL: string; // <<<< REMOVED: No longer used by getDb directly
  MERIDIAN_SECRET_KEY: string;
  CORS_ORIGIN: string;
  ARTICLE_CONTENT_FETCH_QUEUE: Queue;
  // You may also need to add type for the workflows if you're exporting them.
  // E.g., SCRAPE_RSS_FEED: typeof import("cloudflare:workers").Workflow<any, any>;
  // For now, we've converted ScrapeRssFeed to a direct function, so this might not be needed.
  // If `export { ProcessArticles }` is present, it might need `PROCESS_ARTICLES: typeof import("cloudflare:workers").Workflow<any, any>;`
};

// ... (rest of your index.ts, e.g., scheduled handler) ...
// The call to runScrapeRssFeedLogic in scheduled handler now passes env.DATABASE_HYPERDRIVE to it
export default {
  fetch: app.fetch,
  async scheduled({ cron }: ScheduledController, env: Env, ctx: ExecutionContext) {
    if (cron === '4 * * * *') {
      console.log('Scheduled cron: Directly running ScrapeRssFeed logic...');
      await runScrapeRssFeedLogic(env, ctx, { force: true });
      console.log('Scheduled cron: ScrapeRssFeed logic finished.');
    }
    return;
  },
} satisfies ExportedHandler<Env>;

// Ensure this is consistent with your actual file
export { ProcessArticles } from './workflows/processArticles.workflow';