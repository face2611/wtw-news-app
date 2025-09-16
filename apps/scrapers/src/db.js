// C:\Users\phili\meridian\apps\scrapers\src\db.js

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { pgTable, serial, text, timestamp, integer, boolean } from 'drizzle-orm/pg-core'; // Import Drizzle types
import { sql } from 'drizzle-orm';
// Define the schema inline here for maximum isolation
// NOTE: This will temporarily duplicate your schema definition
// but ensures there are no import/export issues from packages/database
export const $sources = pgTable('sources', {
  id: serial('id').primaryKey(),
  url: text('url').notNull().unique(),
  name: text('name').notNull(),
  scrape_frequency: integer('scrape_frequency').notNull().default(2),
  paywall: boolean('paywall').notNull().default(false),
  category: text('category').notNull(),
  lastChecked: timestamp('last_checked', { mode: 'date' }),
});

export const $articles = pgTable('articles', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  url: text('url').notNull().unique(),
  publishDate: timestamp('publish_date', { mode: 'date' }),
  content: text('content'),
  processing_status: text('processing_status').default('Scraped'), // Set default here too
  contentFetchedAt: timestamp('content_fetched_at', { mode: 'date' }),
  geminiProcessedAt: timestamp('gemini_processed_at', { mode: 'date' }),
  run_id: text('run_id'), // <<<< Added run_id here for local schema
  language: text('language'),
  location: text('location'),
  completeness: text('completeness'),
  relevance: text('relevance'),
  summary: text('summary'),
  failReason: text('fail_reason'),
  sourceId: integer('source_id')
    .references(() => $sources.id)
    .notNull(),
  processedAt: timestamp('processed_at', { mode: 'date' }),
  createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});

// Helper function to get DB client
export function getDb(databaseUrl) {
    const queryClient = postgres(databaseUrl);
    return drizzle(queryClient, { schema: { $sources, $articles } }); // Explicitly pass the schemas
}