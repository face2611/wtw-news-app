import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';

import * as schema from './schema';

export const client = (url: string, options?: postgres.Options<{}> | undefined) => postgres(url, options);

export const getDb = (url: string, options?: postgres.Options<{}> | undefined) =>
  drizzle(client(url, options), { schema });
