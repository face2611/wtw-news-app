import { describe, it, expect } from 'vitest';
import { parseRSSFeed } from '../src/lib/parsers';
import { readFileSync } from 'fs';
import path from 'path';

describe('parseRssFeed', () => {
  // helper to load fixtures
  const loadFixture = (filename: string) => readFileSync(path.join(__dirname, 'fixtures', filename), 'utf-8');

  it('handles independant.co.uk feed', async () => {
    const xml = loadFixture('independant_co_uk.xml');
    const result = await parseRSSFeed(xml);
    if (result.isErr()) throw result.error;

    expect(result.value).toHaveLength(100);

    expect(result.value[0].title).toBe(
      'Trump makes good on promise as thousands of JFK assassination files are released: Live updates'
    );
    expect(result.value[0].link).toBe(
      'https://www.independent.co.uk/news/world/americas/us-politics/jfk-files-released-assassination-trump-b2717229.html'
    );
    expect(result.value[0].pubDate).toStrictEqual(new Date('Tue, 18 Mar 2025 23:24:58 GMT'));
  });

  it('handles cn.nytimes.com feed', async () => {
    const xml = loadFixture('cn_nytimes_com.xml');
    const result = await parseRSSFeed(xml);
    if (result.isErr()) throw result.error;

    expect(result.value).toHaveLength(20);

    expect(result.value[0].title).toBe('前高管揭Facebook内幕：配合北京开发审查工具');
    expect(result.value[0].link).toBe('https://cn.nytimes.com/culture/20250318/careless-people-sarah-wynn-williams/');
    expect(result.value[0].pubDate).toStrictEqual(new Date('Tue, 18 Mar 2025 04:59:35 +0800'));
  });

  it('handles ft.com feed', async () => {
    const xml = loadFixture('ft_com.xml');
    const result = await parseRSSFeed(xml);

    if (result.isErr()) throw result.error;

    expect(result.value).toHaveLength(25);

    expect(result.value[0].title).toBe('‘If Trump defies a Supreme Court order, will it matter to markets?’');
    expect(result.value[0].link).toBe('https://www.ft.com/content/2e579290-fc0c-4b88-8703-f0bae45266d9');
    expect(result.value[0].pubDate).toStrictEqual(new Date('Tue, 18 Mar 2025 23:34:47 GMT'));
  });

  it('handles theverge.com feed', async () => {
    const xml = loadFixture('theverge_com.xml');
    const result = await parseRSSFeed(xml);
    if (result.isErr()) throw result.error;

    expect(result.value).toHaveLength(10);

    expect(result.value[0].title).toBe('The Boeing Starliner astronauts returned to Earth today');
    expect(result.value[0].link).toBe(
      'https://www.theverge.com/news/628311/nasa-crew-10-mission-starliner-astronauts-return-spacex'
    );
    expect(result.value[0].pubDate).toStrictEqual(new Date('2025-03-18T18:04:44-04:00'));
  });
});
