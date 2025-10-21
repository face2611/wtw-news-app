// This is the complete and corrected code for apps/scrapers/src/app.ts

import { Hono } from 'hono';
import { trimTrailingSlash } from 'hono/trailing-slash';
import { Env } from './index';
import { getDb, hasValidAuthToken } from './lib/utils';
import openGraph from './routers/openGraph.router';
import reportsRouter from './routers/reports.router';
import { runScrapeRssFeedLogic } from './logic/rssFeed.logic';

// Imports for the /events and /briefs endpoints
import { desc, sql, and, gte, lte, isNotNull, eq, not } from 'drizzle-orm';
import { pgTable, serial, text, date, integer, timestamp } from 'drizzle-orm/pg-core';
import { $sources, $articles } from '@meridian/database'; // We still need these for /events

// THIS IS THE CRITICAL FIX: Define the schema for daily_briefs LOCALLY
export const $dailyBriefs = pgTable('daily_briefs', {
    id: serial('id').primaryKey(),
    briefDate: date('brief_date').notNull().unique(),
    title: text('title'),
    content: text('content').notNull(),
    tldr: text('tldr'),
    run_id: text('run_id'),
    totalArticles: integer('total_articles'),
    totalSources: integer('total_sources'),
    usedArticles: integer('used_articles'),
    usedSources: integer('used_sources'),
    modelAuthor: text('model_author'),
    clusteringParams: text('clustering_params'),
    createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});

export type HonoEnv = { Bindings: Env };

const app = new Hono<HonoEnv>()
  .use(trimTrailingSlash())
  .get('/favicon.ico', async c => c.notFound())

  // THIS IS THE NEW, CORRECT /briefs ENDPOINT
  .get('/briefs', async (c) => {
    try {
      const db = getDb(c.env.DATABASE_URL);
      const briefs = await db.select({
          id: $dailyBriefs.id,
          briefDate: $dailyBriefs.briefDate,
          title: $dailyBriefs.title,
        })
        .from($dailyBriefs)
        .orderBy(desc($dailyBriefs.briefDate))
        .limit(30);
      return c.json(briefs);
    } catch (error: any) {
      console.error('Error fetching briefs:', error.message);
      return c.json({ error: 'Failed to fetch briefs' }, 500);
    }
  })

  .route('/reports', reportsRouter)
  .use('/reports/*', async (c, next) => {
      console.log(`[wtw-production] DEBUG: Incoming request to /reports path: ${c.req.method} ${c.req.url}`);
      await next();
  })
  .route('/openGraph', openGraph)
  .get('/ping', async c => c.json({ pong: true }))
  .get('/events', async c => {
    // This is the existing, working /events endpoint
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