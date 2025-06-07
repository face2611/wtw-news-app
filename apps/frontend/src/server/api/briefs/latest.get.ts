import { $reports, desc, getDb } from '@meridian/database';
import { ensureDate, generateReportSlug } from '~/server/lib/utils';

export default defineEventHandler(async event => {
  const db = getDb(useRuntimeConfig(event).DATABASE_URL);

  const latestReport = await db.query.$reports.findFirst({
    orderBy: desc($reports.createdAt),
    columns: { id: true, createdAt: true, title: true },
  });
  if (latestReport === undefined) {
    throw createError({ statusCode: 404, statusMessage: 'No reports found' });
  }

  return generateReportSlug(ensureDate(latestReport.createdAt));
});
