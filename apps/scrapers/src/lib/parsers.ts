import { Readability } from '@mozilla/readability';
import { parseHTML } from 'linkedom';
import { XMLParser } from 'fast-xml-parser';
import { z } from 'zod';
import { ok, err, Result } from 'neverthrow';
import { cleanString, cleanUrl } from './utils';

const rssFeedSchema = z.object({
  title: z.string().min(1),
  link: z.string(),
  pubDate: z.date().nullable(),
});

export async function parseRSSFeed(xml: string) {
  const safeParser = Result.fromThrowable(
    (xml: string) => new XMLParser({ ignoreAttributes: false, attributeNamePrefix: '@_' }).parse(xml),
    e => (e instanceof Error ? e : new Error(String(e)))
  );

  const parsedXml = safeParser(xml);
  if (parsedXml.isErr()) {
    return err({ type: 'PARSE_ERROR', error: parsedXml.error });
  }

  const result = parsedXml.value;

  // handle various feed structures
  let items = result.rss?.channel?.item || result.feed?.entry || result.item || result['rdf:RDF']?.item || [];

  // handle single item case
  items = Array.isArray(items) ? items : [items];

  const properItems = items.map((item: any) => {
    let title = '';
    let link = '';
    let id = '';
    let pubDateString: string | null = null;

    if (typeof item.title === 'string') {
      title = item.title;
    } else if (typeof item.title === 'object' && item.title['#text']) {
      title = item.title['#text'];
    } else {
      title = 'UNKNOWN';
    }

    if (typeof item.link === 'string') {
      link = item.link;
    } else if (typeof item.link === 'object' && item.link['@_href']) {
      link = item.link['@_href'];
    } else if (typeof item.guid === 'string') {
      link = item.guid;
    } else {
      link = 'UNKNOWN';
    }

    if (typeof item.guid === 'string') {
      id = item.guid;
    } else if (typeof item.guid === 'object' && item.guid['#text']) {
      id = item.guid['#text'];
    } else {
      id = 'UNKNOWN';
    }

    if (typeof item.pubDate === 'string') {
      pubDateString = item.pubDate;
    } else if (typeof item.published === 'string') {
      pubDateString = item.published;
    } else if (typeof item.updated === 'string') {
      pubDateString = item.updated;
    }

    let pubDate: Date | null = null;
    if (pubDateString) {
      pubDate = new Date(pubDateString);
      if (isNaN(pubDate.getTime())) {
        pubDate = null;
      }
    }

    return {
      title: cleanString(title),
      link: cleanUrl(cleanString(link)),
      id: cleanString(id),
      pubDate,
    };
  });

  // standardize the items
  const parsedItems = z.array(rssFeedSchema).safeParse(properItems);
  if (parsedItems.success === false) {
    return err({ type: 'VALIDATION_ERROR', error: parsedItems.error });
  }

  return ok(parsedItems.data);
}

export function parseArticle(opts: { html: string }) {
  const safeReadability = Result.fromThrowable(
    (html: string) => new Readability(parseHTML(html).document).parse(),
    e => (e instanceof Error ? e : new Error(String(e)))
  );

  const articleResult = safeReadability(opts.html);
  if (articleResult.isErr()) {
    return err({ type: 'READABILITY_ERROR', error: articleResult.error });
  }

  // if we can't parse the article or there is no article, not much we can do
  const article = articleResult.value;
  if (article === null || !article.title || !article.textContent) {
    return err({ type: 'NO_ARTICLE_FOUND', error: new Error('No article found') });
  }

  return ok({
    title: article.title,
    text: cleanString(article.textContent),
    publishedTime: article.publishedTime || undefined,
  });
}
