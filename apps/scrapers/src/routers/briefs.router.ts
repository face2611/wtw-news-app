// This is the new, correct, and self-contained code for:
// apps/scrapers/src/routers/briefs.router.ts

import { Hono } from 'hono';
import { HonoEnv } from '../app';
import { getDb } from '../lib/utils';
import { desc } from 'drizzle-orm';
import { pgTable, serial, text, date, integer, timestamp } from 'drizzle-orm/pg-core'; // <-- ADDED IMPORTS

// ADDED: Define the schema for daily_briefs locally, as it's not in the shared package.
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
    createdAt: timestamp('created_at', { mode: 'date' }), // Adjusted for consistency
});

const briefsRouter = new Hono<HonoEnv>()
  .get('/', async (c) => {
    // NOTE: This is a public endpoint and does not check for an auth token.
    // This is different from the /events endpoint.
    try {
      const db = getDb(c.env.DATABASE_URL);
      const briefs = await db.select()
        .from($dailyBriefs)
        .orderBy(desc($dailyBriefs.briefDate))
        .limit(30);

      return c.json(briefs);
    } catch (error: any) {
      console.error('Failed to fetch briefs:', error.message);
      return c.json({ error: 'Failed to fetch briefs from database' }, 500);
    }
  });

export default briefsRouter;