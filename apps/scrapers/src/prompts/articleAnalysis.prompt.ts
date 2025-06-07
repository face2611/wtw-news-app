import { z } from 'zod';

function getArticleAnalysisPrompt(title: string, content: string) {
  return `
<scraped_news_article>
# ${title.trim()}

${content.trim()}
</scraped_news_article>

Extract the following information from this news article and return it as JSON:

{
  "language": "string",  // ISO 639-1 alpha-2 language code of the main language of the article
  "location": "string",  // ISO 3166-1 alpha-3 country code or GLOBAL or N/A
  "completeness": "COMPLETE" | "PARTIAL_USEFUL" | "PARTIAL_USELESS",
  "relevance": "RELEVANT" | "NOT_RELEVANT"
}

Location criteria:
- if we can pin down the article to a specific country, use the ISO 3166-1 alpha-3 country code
- otherwise, use GLOBAL or N/A depending on the article

Completeness criteria:
- COMPLETE: full article with no missing sections, paywalls or cutoffs
- PARTIAL_USEFUL: has some content missing (paywall, subscription nag, truncated) but core info/story is still present
- PARTIAL_USELESS: severely truncated or blocked, just headline/intro with no real substance

Relevance criteria:
- NOT_RELEVANT: celebrity gossip, clickbait, tabloid content, fluff pieces, sponsored content
- RELEVANT: substantive news with factual reporting or meaningful analysis

Summary criteria, if article is RELEVANT and (COMPLETE or PARTIAL_USEFUL):
- extract a terse summary with this structure:
    * HEADLINE: [ultra-concise 5-10 word version of the story]
    * ENTITIES: [key people, orgs, countries involved]
    * EVENT: [what actually happened in 1-2 sentences max]
    * CONTEXT: [why it matters or unique angle in 1 sentence]
- focus on extracting *distinct* details that would help cluster similar stories together. ignore fluff, opinions, and padding
- these summaries will be used for clustering, so surface what makes this story unique or similar to others

Start now and return only the JSON object.
`.trim();
}

export const articleAnalysisSchema = z.object({
  language: z.string().length(2),
  location: z.string().min(3),
  completeness: z.enum(['COMPLETE', 'PARTIAL_USEFUL', 'PARTIAL_USELESS']),
  relevance: z.enum(['RELEVANT', 'NOT_RELEVANT']),
  summary: z
    .object({
      headline: z.string(),
      entities: z.array(z.string()),
      event: z.string(),
      context: z.string(),
    })
    .optional(),
});

export default getArticleAnalysisPrompt;
