import { $reports, eq, and, gte, lte, getDb } from '@meridian/database';
import { ensureDate, formatReportDate } from '~/server/lib/utils';

interface Brief {
  id: number;
  createdAt: Date;
  title: string;
  content: string;
  model_author: string | null;
  totalArticles: number;
  totalSources: number;
  usedSources: number;
  usedArticles: number;
  slug: string;
  date: {
    month: string;
    day: number;
    year: number;
  };
}

export default defineEventHandler(async event => {
  const db = getDb(useRuntimeConfig(event).DATABASE_URL);

  const slug = getRouterParam(event, 'slug');
  if (slug === undefined) {
    throw createError({ statusCode: 400, statusMessage: 'Slug is required' });
  }

  // decode slug & get date
  const date = new Date(slug);
  if (isNaN(date.getTime())) {
    throw createError({ statusCode: 400, statusMessage: 'Invalid slug' });
  }

  // set start/end of the day for date range query
  const startOfDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const endOfDay = new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1);

  // get report created on this day
  const report = await db.query.$reports.findFirst({
    where: and(gte($reports.createdAt, startOfDay), lte($reports.createdAt, endOfDay)),
    columns: {
      id: true,
      createdAt: true,
      title: true,
      content: true,
      model_author: true,
      totalArticles: true,
      totalSources: true,
      usedSources: true,
      usedArticles: true,
    },
  });
  if (report === undefined) {
    throw createError({ statusCode: 404, statusMessage: 'Report not found' });
  }

  return {
    ...report,
    slug,
    date: formatReportDate(ensureDate(report.createdAt)),
  } satisfies Brief;
});
