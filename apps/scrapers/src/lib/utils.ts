import { Env } from '../index';
import { ResultAsync } from 'neverthrow';
import { getDb as getDbFromDatabase } from '@meridian/database';
import { Context } from 'hono';
import { HonoEnv } from '../app';

export function getDb(databaseUrl: string) {
  return getDbFromDatabase(databaseUrl, { prepare: false });
}

export const safeFetch = ResultAsync.fromThrowable(
  (url: string, returnType: 'text' | 'json' = 'text', options: RequestInit = {}) =>
    fetch(url, options).then(res => {
      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      if (returnType === 'text') return res.text();
      return res.json();
    }),
  e => (e instanceof Error ? e : new Error(String(e)))
);

export const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

/**
 * Escape special characters for XML
 */
export function escapeXml(unsafe: string): string {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function cleanString(text: string) {
  return text
    .replace(/[ \t]+/g, ' ') // collapse spaces/tabs
    .replace(/\n\s+/g, '\n') // clean spaces after newlines
    .replace(/\s+\n/g, '\n') // clean spaces before newlines
    .replace(/\n{3,}/g, '\n\n') // keep max 2 consecutive newlines
    .trim(); // clean edges
}

export function cleanUrl(url: string) {
  const u = new URL(url);

  const paramsToRemove = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid'];
  paramsToRemove.forEach(param => u.searchParams.delete(param));

  return u.toString();
}

export function hasValidAuthToken(c: Context<HonoEnv>) {
  const auth = c.req.header('Authorization');
  if (auth === undefined || auth !== `Bearer ${c.env.MERIDIAN_SECRET_KEY}`) {
    return false;
  }
  return true;
}
