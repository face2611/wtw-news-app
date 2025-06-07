import { Hono } from 'hono';
import { z } from 'zod';
import type { HonoEnv } from '../app';
import { $reports, desc, getDb } from '@meridian/database';
import { hasValidAuthToken } from '../lib/utils';
import { zValidator } from '@hono/zod-validator';

const route = new Hono<HonoEnv>()
  .get('/last-report', async c => {
    // check auth token
    const hasValidToken = hasValidAuthToken(c);
    if (!hasValidToken) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    try {
      const report = await getDb(c.env.DATABASE_URL).query.$reports.findFirst({
        orderBy: desc($reports.createdAt),
      });
      if (report === undefined) {
        return c.json({ error: 'No report found' }, 404);
      }

      return c.json(report);
    } catch (error) {
      console.error('Error fetching last report', error);
      return c.json({ error: 'Failed to fetch last report' }, 500);
    }
  })
  .post(
    '/report',
    zValidator(
      'json',
      z.object({
        title: z.string(),
        content: z.string(),
        totalArticles: z.number(),
        totalSources: z.number(),
        usedArticles: z.number(),
        usedSources: z.number(),
        tldr: z.string(),
        createdAt: z.coerce.date(),
        model_author: z.string(),
        clustering_params: z.object({
          umap: z.object({
            n_neighbors: z.number(),
          }),
          hdbscan: z.object({
            min_cluster_size: z.number(),
            min_samples: z.number(),
            epsilon: z.number(),
          }),
        }),
      })
    ),
    async c => {
      // check auth token
      const hasValidToken = hasValidAuthToken(c);
      if (!hasValidToken) {
        return c.json({ error: 'Unauthorized' }, 401);
      }

      const body = c.req.valid('json');

      try {
        await getDb(c.env.DATABASE_URL).insert($reports).values(body);
      } catch (error) {
        console.error('Error inserting report', error);
        return c.json({ error: 'Failed to insert report' }, 500);
      }

      return c.json({ success: true });
    }
  );

export default route;
