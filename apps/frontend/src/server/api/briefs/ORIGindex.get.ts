import { $reports, desc, getDb } from '@meridian/database';
import { ensureDate, formatReportDate, generateReportSlug } from '~/server/lib/utils';

export default defineEventHandler(async event => {
  const db = getDb(useRuntimeConfig(event).DATABASE_URL);

  const reports = await db.query.$reports.findMany({
    orderBy: desc($reports.createdAt),
    columns: { id: true, createdAt: true, title: true },
  });

  // Process reports to add date and slug
  return reports.map(report => {
    const createdAt = ensureDate(report.createdAt);
    return {
      ...report,
      date: formatReportDate(createdAt),
      slug: generateReportSlug(createdAt),
    };
  });
});
