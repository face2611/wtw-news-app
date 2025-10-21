// This is the complete and corrected code for apps/scrapers/src/index.ts

import app from './app';
import { runScrapeRssFeedLogic } from './logic/rssFeed.logic';

// This is the complete and correct type definition for this worker's environment.
// It includes all secrets, variables, and bindings.
export type Env = {
  // Secrets (from Cloudflare Dashboard)
  DATABASE_URL: string;
  MERIDIAN_SECRET_KEY: string;
  CLOUDFLARE_ACCOUNT_ID: string;
  CLOUDFLARE_BROWSER_RENDERING_API_TOKEN: string;
  GOOGLE_API_KEY: string;
  GOOGLE_BASE_URL: string;
  
  // Variables (from Cloudflare Dashboard or wrangler.toml)
  CORS_ORIGIN: string;

  // Bindings (from wrangler.toml)
  ARTICLE_CONTENT_FETCH_QUEUE: Queue;
  BROWSER: Fetcher;
};

export default {
  fetch: app.fetch,

  async scheduled(controller: ScheduledController, env: Env, ctx: ExecutionContext) {
    if (controller.cron === '4 * * * *') {
      console.log('Scheduled cron: Running ScrapeRssFeed logic...');
      await runScrapeRssFeedLogic(env, ctx, { force: true });
      console.log('Scheduled cron: ScrapeRssFeed logic finished.');
    }
    return;
  },
} satisfies ExportedHandler<Env>;