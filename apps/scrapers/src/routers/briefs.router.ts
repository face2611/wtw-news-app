// This is the content for the new file: apps/scrapers/src/routers/briefs.router.ts
import { Hono } from 'hono';
import { HonoEnv } from '../app';
import { getDb } from '../lib/utils';
import { $dailyBriefs } from '@meridian/database';
import { desc } from 'drizzle-orm';

const briefsRouter = new Hono<HonoEnv>()
  .get('/', async (c) => {
    try {
      const db = getDb(c.env.DATABASE_URL);
      const briefs = await db.select()
        .from($dailyBriefs)
        .orderBy(desc($dailyBriefs.briefDate))
        .limit(30); // Get the last 30 briefs

      return c.json(briefs);
    } catch (error) {
      console.error('Failed to fetch briefs:', error);
      return c.json({ error: 'Failed to fetch briefs from database' }, 500);
    }
  });

export default briefsRouter;