// Reverting to the clean version without the /briefs endpoint.

import { Hono } from 'hono';
import { trimTrailingSlash } from 'hono/trailing-slash';
import { Env } from './index';
import { getDb, hasValidAuthToken } from './lib/utils';
import openGraph from './routers/openGraph.router';
import reportsRouter from './routers/reports.router';
import { runScrapeRssFeedLogic } from './logic/rssFeed.logic';

// Imports for the /events endpoint
import { and, gte, lte, isNotNull, eq, not } from 'drizzle-orm';
import { $sources, $articles } from '@meridian/database';

export type HonoEnv = { Bindings: Env };

const app = new Hono<HonoEnv>()
  .use(trimTrailingSlash())
  .get('/favicon.ico', async c => c.notFound())
  // The broken /briefs endpoint has been removed.
  .route('/reports', reportsRouter)
  .use('/reports/*', async (c, next) => {
      console.log(`[wtw-production] DEBUG: Incoming request to /reports path: ${c.req.method} ${c.req.url}`);
      await next();
  })
  .route('/openGraph', openGraph)
  .get('/ping', async c => c.json({ pong: true }))
  .get('/events', async c => {
    const hasValidToken = hasValidAuthToken(c);
    if (!hasValidToken) {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    const dateParam = c.req.query('date');
    let endDate: Date;
    if (dateParam) {
      endDate = new Date(`${dateParam}T07:00:00Z`);
      if (isNaN(endDate.getTime())) {
        return c.json({ error: 'Invalid date format. Please use yyyy-mm-dd' }, 400);
      }
    } else {
      endDate = new Date();
      endDate.setUTCHours(7, 0, 0, 0);
    }
    const startDate = new Date(endDate.getTime() - 30 * 60 * 60 * 1000);
    const db = getDb(c.env.DATABASE_URL);
    const allSources = await db.select({ id: $sources.id, name: $sources.name }).from($sources);
    let events = await db.select().from($articles).where(
        and(
          isNotNull($articles.location),
          gte($articles.publishDate, startDate),
          lte($articles.publishDate, endDate),
          eq($articles.relevance, 'RELEVANT'),
          not(eq($articles.completeness, 'PARTIAL_USELESS')),
          isNotNull($articles.summary)
        )
      );
    const response = { sources: allSources, events, dateRange: { startDate: startDate.toISOString(), endDate: endDate.toISOString() } };
    return c.json(response);
  })
  .get('/trigger-rss', async c => {
    const token = c.req.query('token');
    if (token !== c.env.MERIDIAN_SECRET_KEY) {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    try {
      console.log('API Trigger: Directly running ScrapeRssFeed logic...');
      await runScrapeRssFeedLogic(c.env, c.executionCtx, { force: true });
      console.log('API Trigger: ScrapeRssFeed logic finished.');
      return c.json({ success: true });
    } catch (error: any) {
      console.error('API Trigger: Error running ScrapeRssFeed logic:', error);
      return c.json({ error: error.message || 'Internal Server Error' }, 500);
    }
  });

export default app;