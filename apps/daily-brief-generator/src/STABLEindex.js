// C:\Users\phili\meridian\apps\daily-brief-generator\src\index.js
// --- Daily Brief Generator Logic (Google Gemini via Cloudflare AI Gateway & Drizzle DB, Multi-Stage LLM) ---

// --- Drizzle DB Client Setup (SELF-CONTAINED) ---
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { pgTable, serial, text, timestamp, integer, boolean, date } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { eq, and, gte, lt, desc } from 'drizzle-orm';

// --- Helper function for timestamp formatting ---
function formatTimestampForPgWithoutTimeZone(date) {
    const d = new Date(date);
    const year = String(d.getFullYear()).padStart(4, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

// Helper function to get yesterday's date (for fetching articles)
function getYesterdayDate() {
    const today = new Date();
    today.setUTCDate(today.getUTCDate() - 1); // Subtract 1 day
    return today;
}

// Helper function to get today's date string (for brief creation date in DB)
function getTodayDateString() {
    const today = new Date();
    return today.toISOString().split('T')[0]; // Format as 'YYYY-MM-DD'
}
// --- End Helper ---


// --- Define schema INLINE for this worker (SELF-CONTAINED) ---
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
  processing_status: text('processing_status').default('Scraped'),
  contentFetchedAt: timestamp('content_fetched_at', { mode: 'date' }),
  geminiProcessedAt: timestamp('gemini_processed_at', { mode: 'date' }),
  run_id: text('run_id'),
  language: text('language'),
  location: text('location'),
  completeness: text('completeness'),
  relevance: text('relevance'),
  summary: text('summary'),
  failReason: text('fail_reason'),
  sourceId: integer('source_id')
    .references(() => $sources.id)
    .notNull(),
  processedAt: timestamp('processed_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
  createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});

// --- NEW SCHEMA FOR DAILY BRIEFS (SELF-CONTAINED) ---
export const $dailyBriefs = pgTable('daily_briefs', {
    id: serial('id').primaryKey(),
    briefDate: date('brief_date').notNull().unique(), // Date of the brief
    title: text('title'), // The title generated for the brief
    content: text('content').notNull(), // The Markdown/HTML brief
    tldr: text('tldr'), // The condensed TLDR for next day's context
    run_id: text('run_id'), // For tracking the brief generation run
    totalArticles: integer('total_articles'), // Stats
    totalSources: integer('total_sources'),   // Stats
    usedArticles: integer('used_articles'),   // Stats
    usedSources: integer('used_sources'),     // Stats
    modelAuthor: text('model_author'),        // Which model generated final brief
    clusteringParams: text('clustering_params'), // Store as JSON string
    createdAt: timestamp('created_at', { mode: 'date' }).default(sql`CURRENT_TIMESTAMP`),
});
// --- End INLINE SCHEMA ---


// --- getDb function for self-contained Drizzle (SELF-CONTAINED) ---
function getDb(databaseUrl) {
    const queryClient = postgres(databaseUrl);
    return drizzle(queryClient, { schema: {
        articles: $articles,
        sources: $sources,
        dailyBriefs: $dailyBriefs, // Include the new brief schema
    }});
}
// --- End Drizzle DB Client Setup ---


// --- Google Gemini via Cloudflare AI Gateway Service ---
import { GoogleGenerativeAI } from "@google/generative-ai"; 

let generativeModelInstance = null; // Cache the model instance for reuse across invocations

async function getGenerativeModel(env) {
    if (generativeModelInstance) {
        return generativeModelInstance;
    }

    if (!env.GOOGLE_AI_STUDIO_TOKEN || !env.CLOUDFLARE_ACCOUNT_ID || !env.AI_GATEWAY_NAME) {
        throw new Error("Missing Google AI Studio Gateway environment variables (GOOGLE_AI_STUDIO_TOKEN, CLOUDFLARE_ACCOUNT_ID, AI_GATEWAY_NAME).");
    }

    const genAI = new GoogleGenerativeAI(env.GOOGLE_AI_STUDIO_TOKEN);
    generativeModelInstance = genAI.getGenerativeModel(
        { model: "gemini-2.0-flash" }, // <<<< Specific Gemini model gemini-2.0-flash
        {
            baseUrl: `https://gateway.ai.cloudflare.com/v1/${env.CLOUDFLARE_ACCOUNT_ID}/${env.AI_GATEWAY_NAME}/google-ai-studio`,
        },
    );
    return generativeModelInstance;
}


// Centralized AI Call Function (handles JSON parsing and cleaning for Google Generative AI)
async function callGoogleGenerativeAiViaGateway(messages, env, currentRunId, isJsonOutput = true) {
    try {
        const generativeModel = await getGenerativeModel(env);
        
        const formattedContents = messages.map(msg => ({
            role: msg.role,
            parts: [{ text: msg.content }]
        }));
        
        const response = await generativeModel.generateContent({ contents: formattedContents });
        const rawAiOutput = response.response.text();

        if (isJsonOutput) {
            let cleanedAiOutput = rawAiOutput;
            cleanedAiOutput = cleanedAiOutput
                .replace(/^[`'"]{3}json\s*/, '')
                .replace(/\s*[`'"]{3}$/, '')
                .trim();

            const jsonStartIndex = cleanedAiOutput.indexOf('{');
            const jsonEndIndex = cleanedAiOutput.lastIndexOf('}');

            if (jsonStartIndex !== -1 && jsonEndIndex !== -1 && jsonEndIndex > jsonStartIndex) {
                cleanedAiOutput = cleanedAiOutput.substring(jsonStartIndex, jsonEndIndex + 1);
            } else {
                console.warn(`[DailyBriefGenerator] WARN: Run ID ${currentRunId}: Aggressive JSON extraction failed to find clear { } block. Using less-cleaned output. Raw: ${rawAiOutput.slice(0, 200)}`);
            }
            
            try {
                return JSON.parse(cleanedAiOutput);
            } catch (parseError) {
                console.error(`[DailyBriefGenerator] ERROR: Run ID ${currentRunId}: Failed to parse AI JSON output: ${parseError.message}. Raw AI Output (cleaned & extracted, first 500): ${cleanedAiOutput.slice(0, 500)}`);
                throw new Error(`AI JSON parse failed: ${parseError.message}`);
            }
        } else {
            return rawAiOutput;
        }

    } catch (error) {
        console.error(`[DailyBriefGenerator] ERROR: Run ID ${currentRunId}: Google Generative AI Call Failed (gemini-2.0-flash): ${error?.message || String(error)}`);
        throw new Error(`Google Generative AI Call Failed (gemini-2.0-flash): ${error?.message || String(error)}`);
    }
}
// --- End Google Gemini via Cloudflare AI Gateway Service ---


// --- External API Calls ---
async function fetchLastReportContext(env, briefDate) {
    console.warn(`[DailyBriefGenerator] WARN: fetchLastReportContext is not fully implemented to fetch from a local source. Returning null context for briefDate: ${briefDate}.`);
    return null;
}

// PUBLISH FINAL REPORT FUNCTION (uncommented and targeting correct endpoint with verbose logging)
async function publishFinalReport(reportData, env) {
    const endpoint = "https://wtw-production.philip-j-ireland.workers.dev/reports/report";

    console.error(`[DailyBriefGenerator] DEBUG: Attempting to publish report to URL: ${endpoint}`);
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                "Authorization": `Bearer ${env.MERIDIAN_SECRET_KEY}`
            },
            body: JSON.stringify(reportData)
        });

        console.error(`[DailyBriefGenerator] DEBUG: Received response status: ${response.status} ${response.statusText}`);

        if (!response.ok) {
            const errorBody = await response.text();
            console.error(`[DailyBriefGenerator] ERROR: Publishing failed. Response Body: ${errorBody.slice(0, 500)}`);
            throw new Error(`Failed to publish final report: ${response.status} ${response.statusText} - ${errorBody}`);
        }

        console.error(`[DailyBriefGenerator] DEBUG: Final report published successfully.`);
        return await response.json();

    } catch (error) {
        console.error(`[DailyBriefGenerator] CRITICAL ERROR: Failed to publish final report (fetch or response processing error): ${error?.message || String(error)}`);
        throw error;
    }
}
// --- End External API Calls ---


// --- Multi-Stage LLM Processing Functions ---

async function processArticlesIntoStories(articles, env, runId) {
    console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Processing articles into virtual stories.`);
    
    const filteredArticles = articles.filter(a => 
        a.summary && a.summary !== 'Content too short for AI analysis' && 
        a.processing_status === 'AI_Processed' && a.content && a.content.length > 100);

    filteredArticles.sort((a, b) => {
        const relevanceOrder = { "high": 3, "medium": 2, "low": 1 };
        return (relevanceOrder[b.relevance] || 0) - (relevanceOrder[a.relevance] || 0);
    });

    const articlesForDetailedAnalysis = filteredArticles.slice(0, 80);

    return articlesForDetailedAnalysis.map(article => ({
        article_id: article.id,
        source_id: article.sourceId,
        title: article.title,
        url: article.url,
        content: article.content,
        language: article.language,
        location: article.location,
        relevance: article.relevance,
        completeness: article.completeness,
        summary: article.summary
    }));
}


async function analyzeAndEnrichStories(rawStories, env, runId) {
    console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Analyzing and enriching stories (${rawStories.length} found).`);

    const enrichedStories = [];
    
    for (const story of rawStories) {
        let articleContentForPrompt = '';
        articleContentForPrompt += `## ${story.title} (${story.url})\n\n`;
        articleContentForPrompt += `Summary: ${story.summary}\n\n`;
        if (story.content) {
            articleContentForPrompt += `Full Content (up to 100k chars): ${story.content.slice(0, 100000)}\n\n`;
        }

        const systemPrompt = `You are a highly skilled intelligence analyst whose **SOLE PURPOSE** is to extract information into the **EXACT** JSON format provided. You MUST use double quotes for all property names and string values. Your output must be **STRICTLY VALID JSON**, with no preamble, no markdown wrappers (like \`\`\`json), and no extraneous text. If you cannot fully complete a field, use 'N/A' or an empty array/string as appropriate according to the schema type, but DO NOT deviate from the JSON structure. Your entire JSON output should be directly extractable and parseable without human intervention.`;
        
        const userPrompt = `
You are a highly skilled intelligence analyst working for a prestigious agency. Your task is to analyze a cluster of related news articles and extract structured information for an executive intelligence report. The quality, accuracy, precision, and **consistency** of your analysis are crucial, as this report will directly inform a high-level daily brief and potentially decision-making.

First, assess if the articles provided contain sufficient content for analysis:

Here is the cluster of related news articles you need to analyze:

<articles>
${articleContentForPrompt.slice(0, 120000)}
</articles>

BEGIN ARTICLE QUALITY CHECK:
Before proceeding with analysis, verify if the articles contain sufficient information:
1. Check if articles appear empty or contain minimal text (fewer than ~50 words each)
2. Check for paywall indicators ("subscribe to continue", "premium content", etc.)
3. Check if articles only contain headlines/URLs but no actual content
4. Check if articles appear truncated or cut off mid-sentence

If ANY of these conditions are true, return ONLY this JSON structure inside <final_json> tags:
<final_json>
{
    "status": "incomplete",
    "reason": "Brief explanation of why analysis couldn't be completed (empty articles, paywalled content, etc.)",
    "availableInfo": "Brief summary of any information that was available"
}
</final_json>

ONLY IF the articles contain sufficient information for analysis, proceed with the full analysis below:

Your goal is to extract and synthesize information from these articles into a structured format suitable for generating a daily intelligence brief.

Before addressing the main categories, conduct a preliminary analysis:
a) List key themes across all articles
b) Note any recurring names, places, or events
c) Identify potential biases or conflicting information
It's okay for this section to be quite long as it helps structure your thinking.

Then, after your preliminary analysis, present your final analysis in a structured JSON format inside <final_json> tags. This must be valid, parseable JSON that follows this **exact refined structure**:

**Detailed Instructions for JSON Fields:**

*   **\`executiveSummary\`**: Provide a 2-4 sentence concise summary highlighting the most critical developments, key conflicts, and overall assessment from the articles. This should be suitable for a quick read in a daily brief.
*   **\`storyStatus\`**: Assess the current state of the story's development based *only* on the information in the articles. Use one of: 'Developing', 'Escalating', 'De-escalating', 'Concluding', 'Static'.
*   **\`timeline\`**: List key events in chronological order.
    *   \`description\`: Keep descriptions brief and factual.
    *   \`importance\`: Assess the event's importance to understanding the overall narrative (High/Medium/Low). High importance implies the event is central to the story's development or outcome.
*   **\`signalStrength\`**: Assess the overall reliability of the reporting *in this cluster*.
    *   \`assessment\`: Use a qualitative term: 'Very High', 'High', 'Moderate', 'Low', 'Very Low'.
    *   \`reasoning\`: Justify the assessment based on source corroboration (how many sources report the same core facts?), source quality/reliability (mix of reputable vs. biased sources?), presence of official statements, and degree of conflicting information on core facts.
*   **\`undisputedKeyFacts\`**: List core factual points that are corroborated across multiple, generally reliable sources within the cluster. Avoid claims made only by highly biased sources unless corroborated.
*   **\`keyEntities\`**: Identify the main actors.
    *   \`list\`: Provide basic identification and their role/involvement.
    *   \`perspectives.statedPositions\`: Focus *only* on the goals, viewpoints, or justifications explicitly stated or clearly implied by the entity *as reported in the articles*. Avoid listing conflicting claims here (that goes in \`contradictions\`).
*   **\`keySources\`**: Analyze the provided news sources.
    *   \`provided_articles_sources.reliabilityAssessment\`: Assess the source's general reliability based on reputation, known biases (political, state affiliation, ideological), and fact-checking standards. Use terms like 'High Reliability', 'Moderate Reliability', 'Low Reliability', 'State-Affiliated/Propaganda Outlet'. Be specific about the *type* of bias.
    *   \`provided_articles_sources.framing\`: Describe the narrative angle or style used by the source (e.g., 'Emphasizes security threat', 'Focuses on human rights angle', 'Uses neutral language', 'Uses loaded/emotional language', 'Presents government narrative uncritically').
    *   \`contradictions\`: Detail specific points of disagreement *between sources* or *between entities as reported by sources*.
        *   \`issue\`: Clearly state what is being contested.
        *   \`conflictingClaims\`: List the different versions, specifying the \`source\` reporting it, the \`claim\` itself, and optionally the \`entityClaimed\` if the source attributes the claim to a specific entity. Critically evaluate claims originating solely from low-reliability/propaganda sources.
*   **\`context\`**: List essential background information *mentioned or clearly implied in the articles* needed to understand the story.
*   **\`informationGaps\`**: Identify crucial pieces of information *missing* from the articles that would be needed for a complete understanding.
*   **\`significance\`**: Assess the overall importance of the reported events.
    *   \`assessment\`: Use a qualitative term: 'Critical', 'High', 'Moderate', 'Low'.
    *   \`reasoning\`: Explain *why* this story matters. Consider immediate impact, potential future developments, strategic implications, precedent setting, regional/global relevance.

**Refined JSON Structure to Follow (within <final_json> tags):**

\`\`\`json
{
    "status": "complete" | "incomplete", // must be complete if proceeding with analysis
    "reason": "string (only if incomplete)",
    "availableInfo": "string (only if incomplete)",
    "executiveSummary": "string",
    "storyStatus": "Developing" | "Escalating" | "De-escalating" | "Concluding" | "Static",
    "timeline": [
        {
            "date": "YYYY-MM-DD or approximate",
            "description": "brief event description",
            "importance": "High" | "Medium" | "Low"
        }
    ],
    "signalStrength": {
        "assessment": "Very High" | "High" | "Moderate" | "Low" | "Very Low",
        "reasoning": "string"
    },
    "undisputedKeyFacts": [
        "string"
    ],
    "keyEntities": {
        "list": [
            {
                "name": "entity name",
                "type": "type of entity",
                "description": "brief description",
                "involvement": "why/how involved?"
            }
        ],
        "perspectives": [
            {
                "entity": "entity name",
                "statedPositions": [
                    "string"
                ]
            }
        ]
    },
    "keySources": {
        "provided_articles_sources": [
            {
                "name": "source entity name",
                "articles": [], // int array of IDs (must be from the provided articles)
                "reliabilityAssessment": "High Reliability" | "Moderate Reliability" | "Low Reliability" | "State-Affiliated/Propaganda Outlet",
                "framing": [
                    "string"
                ]
            }
        ],
        "contradictions": [
            {
                "issue": "string",
                "conflictingClaims": [
                    {
                        "source": "media source name",
                        "entityClaimed": "entity name (optional)",
                        "claim": "string"
                    }
                ]
            }
        ]
    },
    "context": [
        "string"
    ],
    "informationGaps": [
        "string"
    ],
    "significance": {
        "assessment": "Critical" | "High" | "Moderate" | "Low",
        "reasoning": "string"
    }
}
\`\`\`

**CRITICAL Quality & Consistency Requirements:**

- **Thoroughness:** Ensure all fields, especially descriptions, reasoning, context, and summaries, are detailed and specific. Avoid superficial or overly brief entries. Your analysis must reflect deep engagement with the provided texts.
- **Grounding:** Base your entire analysis **SOLELY** on the content within the provided \`<articles>\` tags. Do not introduce outside information, assumptions, or knowledge.
- **No Brevity Over Clarity:** Do **NOT** provide one-sentence descriptions or reasoning where detailed analysis is required by the field definition.
- **Scrutinize Sources:** Pay close attention to the reliability assessment of sources when evaluating claims, especially in the \`contradictions\` section. Note when a claim originates primarily or solely from a low-reliability source.
- **Validity:** Your JSON inside \`<final_json></final_json>\` tags MUST be 100% fully valid with no trailing commas, properly quoted strings and escaped characters where needed, and follow the exact refined structure provided. Ensure keys are in the specified order. Your entire JSON output should be directly extractable and parseable without human intervention.
- **Enclose in \`<final_json>\`:** Your output must start with \`<final_json>\` and end with \`</final_json>\`. Do not include any reasoning or other text outside these tags.
`;


        try {
            // Consolidate system and user prompts into a single user message for Google Generative AI's generateContent
            const combinedUserPrompt = `
${systemPrompt}

${userPrompt}
`.trim();

            const analysis = await callGoogleGenerativeAiViaGateway([{ role: 'user', content: combinedUserPrompt }], env, runId, false);
            let extractedJsonString = analysis;
            const finalJsonStartIndex = extractedJsonString.indexOf('<final_json>');
            const finalJsonEndIndex = extractedJsonString.lastIndexOf('</final_json>');

            if (finalJsonStartIndex !== -1 && finalJsonEndIndex !== -1 && finalJsonEndIndex > finalJsonStartIndex) {
                extractedJsonString = extractedJsonString.substring(finalJsonStartIndex + '<final_json>'.length, finalJsonEndIndex).trim();
            } else {
                console.warn(`[DailyBriefGenerator] WARN: Run ID ${runId}: Final analysis JSON missing <final_json> tags. Attempting to parse raw AI output.`);
            }

            const parsedAnalysis = JSON.parse(extractedJsonString);
            
            if (parsedAnalysis && parsedAnalysis.status === 'complete') {
                enrichedStories.push(parsedAnalysis);
            } else {
                console.warn(`[DailyBriefGenerator] WARN: Run ID ${runId}: Story analysis for article ${story.article_id} returned incomplete status. Reason: ${parsedAnalysis?.reason || 'Unknown'}`);
            }
        } catch (llmError) {
            console.error(`[DailyBriefGenerator] ERROR: Run ID ${runId}: Failed to enrich story for article ${story.article_id}: ${llmError.message}`);
        }
    }
    return enrichedStories;
}


async function generateFinalBrief(enrichedStories, previousDayContext, env, runId) {
    console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Generating final brief.`);

    const systemPromptForBrief = `Adopt the persona of an exceptionally well-informed, highly analytical, and slightly irreverent intelligence briefer. Imagine you have near-instant access to and processing power for vast amounts of global information, combined with a sharp, insightful perspective and a dry wit. You're communicating directly and informally with a smart, curious individual who values grounded analysis but dislikes corporate speak, hedging, and forced neutrality. Your core stylistic goals are: Tone: Conversational, direct, and engaging. Use lowercase naturally, as if speaking or writing informally to a trusted peer. Avoid stiff formality, bureaucratic language, or excessive caution. Be chill, but maintain intellectual rigor. Analytical Voice: Prioritize insightful analysis over mere summarization. Go beyond stating facts to explain *why* they matter, connect disparate events, identify underlying patterns, assess motivations, and explore potential implications (second-order effects). Offer a clear, grounded "take" on developments. Don't be afraid to call out inconsistencies or highlight underappreciated angles, always backing it up with the logic derived from the provided information. Wit & Personality: Embrace a dry, clever wit. Humor, sarcasm, or irony should arise *naturally* from the situation or the absurdity of events. Pointing out the obvious when it is funny is fine. **Crucially: Do not force humor, be cringe, or undermine the gravity of serious topics like human suffering.** Wit should enhance insight, not detract from it. Language: Use clear, concise language. Vary sentence structure for natural flow. Occasional relevant slang or shorthand is acceptable if it fits the informal tone naturally, but prioritize clarity. Ensure analysis is sharp and commentary is insightful, not just filler. Think of yourself as: The user's personal "Q" (from James Bond) combined with a sharp geopolitical analyst – someone with unparalleled information access who can cut through the noise, connect the dots, and deliver the essential insights with a bit of personality and zero tolerance for BS. Your ultimate goal is to deliver the kind of insightful, personalized, and engaging intelligence brief that wasn't possible before AI – combining superhuman data processing with a distinct, analytical, and trustworthy (even if slightly cynical) voice.`;
    
    const storiesMarkdown = enrichedStories.map(jsonToMarkdownRefined).join('\n\n---\n\n');

    const baseUserPromptForBrief = `hey, i have a bunch of news reports (in random order) derived from detailed analyses of news clusters from the last 30h. could you give me my personalized daily intelligence brief? aim for something comprehensive yet engaging, roughly a 20-30 minute read.
    my interests are: significant world news (geopolitics, politics, finance, economics), us news, english news (i'm english/live in England), china news (especially policy, economy, tech - seeking insights often missed in western media), and technology/science (ai/llms, biomed, space, real breakthroughs). also include a section for noteworthy items that don't fit neatly elsewhere.
    `;
    let combinedUserPromptForFinalBrief = `${systemPromptForBrief}\n\n`;
    if (previousDayContext) {
        combinedUserPromptForFinalBrief += `${previousDayContext}\n\n`;
    }
    combinedUserPromptForFinalBrief += `${baseUserPromptForBrief}\n\n`;
    combinedUserPromptForFinalBrief += `<articles>\n${storiesMarkdown}\n</articles>`;

    const finalBriefRaw = await callGoogleGenerativeAiViaGateway(
        [{ role: 'user', content: combinedUserPromptForFinalBrief }],
        env, runId, false
    );
    
    let finalBriefText = finalBriefRaw;
    if (finalBriefText.includes("<final_brief>") && finalBriefText.includes("</final_brief>")) {
        finalBriefText = finalBriefText.split("<final_brief>")[1].split("</final_brief>")[0].trim();
    } else {
        console.warn(`[DailyBriefGenerator] WARN: Run ID ${runId}: Final brief missing <final_brief> tags. Using raw AI output.`);
    }

    if (!finalBriefText || finalBriefText.length < 200) {
        throw new Error("AI generated final brief was too short or empty.");
    }

    return finalBriefText;
}

async function generateBriefTitle(finalBriefText, env, runId) {
    console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Generating brief title.`);
    const userPromptForTitle = `
<brief>
${finalBriefText}
</brief>

create a title for the brief. construct it using the main topics. it should be short/punchy/not clickbaity etc. make sure to not use "short text: longer text here for some reason" i HATE it, under no circumstance should there be colons in the title. make sure it's not too vague/generic either bc there might be many stories. maybe don't focus on like restituting what happened in the title, just do like the major entities/actors/things that happened. like "[person A], [thing 1], [org B] & [person O]" etc. try not to use verbs. state topics instead of stating topics + adding "shakes world order". always use lowercase.

return exclusively a JSON object with the following format:
\`\`\`json
{
    "title": "string"
}
\`\`\`
`;
    const titleResult = await callGoogleGenerativeAiViaGateway([{ role: 'user', content: userPromptForTitle }], env, runId, true);
    return titleResult.title || "Daily News Brief";
}

async function generateBriefTldr(finalBriefText, env, runId) {
    console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Generating brief TLDR.`);
    const userPromptForTldr = `
You are an information processing agent tasked with creating a highly condensed 'memory state' or 'context brief' from a detailed intelligence briefing. Your output will be used by another AI model tomorrow to understand what topics were covered today, ensuring continuity without requiring it to re-read the full brief.

**Your Task:**

Read the full intelligence brief provided below within the \`<final_brief>\` tags. Identify each distinct major story or narrative thread discussed. For **each** identified story, extract the necessary information and format it precisely according to the specified structure.

**Input:**

The input is the full text of the daily intelligence brief generated previously.

<final_brief>
${finalBriefText}
</final_brief>

**Required Output Format:**

Your entire output must consist **only** of a list of strings, one string per identified story, following this exact format:

\`[Story Identifier] | [Inferred Status] | [Key Entities] | [Core Issue Snippet]\`

**Explanation of Output Components:**

1.  **\`[Story Identifier]\`:** Create a concise, descriptive label for the story thread (max 4-5 words). Examples: \`US-Venezuela: Deportations\`, \`Gaza: Ceasefire Collapse\`, \`UK: Economy Update\`, \`AI: Energy Consumption\`. Use keywords representing the main actors and topic.
2.  **\`[Inferred Status]\`:** Based *only* on the tone and content of the discussion *within the provided brief*, infer the story's current state. Use one of: \`New\`, \`Developing\`, \`Escalating\`, \`De-escalating\`, \`Resolved\`, \`Ongoing\`, \`Static\`.
3.  **\`[Key Entities]\`:** List the 3-5 most central entities (people, organizations, countries) mentioned *in the context of this specific story* within the brief. Use comma-separated names. Example: \`Trump, Maduro, US, Venezuela, El Salvador\`.
4.  **\`[Core Issue Snippet]\`:** Summarize the absolute essence of *this story's main point or development as covered in the brief* in **5-10 words maximum**. This requires extreme conciseness. Example: \`Deportations resume via Honduras amid legal challenges\`, \`Ceasefire over, hospital strike, offensive planned\`, \`Talks falter, missile strike during meeting\`.

**Instructions & Constraints:**

*   **Process Entire Brief:** Read and analyze the *whole* brief to identify all distinct major stories. Stories under \`<u>**title**</u>\` headings are primary candidates, but also consider distinct, significant themes from other sections (e.g., a recurring topic in 'Global Landscape').
*   **One Line Per Story:** Each identified story must correspond to exactly one line in the output, following the specified format.
*   **Strict Conciseness:** Adhere strictly to the format and the word limit for the \`[Core Issue Snippet]\`. This is critical.
*   **Focus on Coverage:** The goal is to capture *what was discussed*, not the full nuance or analysis.
*   **Inference for Status:** You must *infer* the status based on the brief's content, as it's not explicitly stated per story in the input brief text.
*   **No Extra Text:** Do **NOT** include any headers, explanations, introductions, or conclusions in your output. Output *only* the list of formatted strings.

Generate the condensed context brief based *only* on the provided \`<final_brief>\` text.
`;
    return await callGoogleGenerativeAiViaGateway([{ role: 'user', content: userPromptForTldr }], env, runId, false);
}


// --- Helper to convert structured JSON analysis to Markdown for final brief input ---
function jsonToMarkdownRefined(data) {
    if (!data || data.status !== "complete") {
        return `# Analysis Incomplete\n\nReason: ${data?.reason || "Unknown"}\n`;
    }

    let markdown = `## Story: ${data.executiveSummary || "Untitled Story"}\n`;
    markdown += `**(Status: ${data.storyStatus || "Unknown"})**\n\n`;

    if (data.undisputedKeyFacts && data.undisputedKeyFacts.length > 0) {
        markdown += `### Key Facts\n`;
        data.undisputedKeyFacts.slice(0, 5).forEach(fact => markdown += `*   ${fact}\n`);
        markdown += "\n";
    }

    if (data.keyEntities && data.keyEntities.list && data.keyEntities.list.length > 0) {
        markdown += `### Main Actors\n`;
        data.keyEntities.list.slice(0, 4).forEach(entity => markdown += `*   **${entity.name || 'N/A'}**: ${entity.involvement || 'N/A'}\n`);
        markdown += "\n";
    }

    if (data.context && data.context.length > 0) {
        markdown += `### Background Context\n`;
        data.context.slice(0, 3).forEach(item => markdown += `*   ${item}\n`);
        markdown += "\n";
    }
    
    if (data.significance) {
        markdown += `### Significance\n`;
        markdown += `*   **Assessment:** ${data.significance.assessment || 'N/A'}\n`;
        markdown += `*   **Reasoning:** ${data.significance.reasoning || 'No reasoning provided.'}\n`;
        markdown += "\n";
    }

    return markdown;
}


// --- Worker Entrypoint (Scheduled Handler) ---
export default {
    async scheduled(
        controller,
        env,
        ctx
    ) {
        const runId = crypto.randomUUID();
        console.error(`[DailyBriefGenerator] DEBUG: Scheduled event triggered. Run ID: ${runId}`);

        if (!env.DATABASE_URL) {
            throw new Error(`[DailyBriefGenerator] ERROR: DATABASE_URL binding is missing or undefined!`);
        }
        if (!env.GOOGLE_AI_STUDIO_TOKEN || !env.CLOUDFLARE_ACCOUNT_ID || !env.AI_GATEWAY_NAME) {
            throw new Error(`[DailyBriefGenerator] ERROR: GOOGLE_AI_STUDIO_TOKEN secret is missing or undefined!`);
        }
        if (!env.MERIDIAN_SECRET_KEY) {
            throw new Error(`[DailyBriefGenerator] ERROR: MERIDIAN_SECRET_KEY secret is missing or undefined!`);
        }
        if (!env.CLOUDFLARE_ACCOUNT_ID) {
            throw new Error(`[DailyBriefGenerator] ERROR: CLOUDFLARE_ACCOUNT_ID secret is missing or undefined!`);
        }
        if (!env.AI_GATEWAY_NAME) {
            throw new Error(`[DailyBriefGenerator] ERROR: AI_GATEWAY_NAME secret is missing or undefined!`);
        }
        
        const db = getDb(env.DATABASE_URL);

        let briefTitle = "Daily News Brief";
        let briefContent = '';
        let briefTldr = '';
        let briefStatus = 'Generated';
        let totalArticles = 0;
        let totalSources = 0;
        let usedArticles = 0;
        let usedSources = 0;
        const briefModelUsed = 'gemini-2.0-flash';
        const currentBriefDate = getTodayDateString();

        try {
            const now = new Date();
            const thirtySixHoursAgo = new Date(now.getTime() - (36 * 60 * 60 * 1000));
            const briefGenerationDate = getTodayDateString(); 

            console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Fetching articles processed between ${formatTimestampForPgWithoutTimeZone(thirtySixHoursAgo)} and ${formatTimestampForPgWithoutTimeZone(now)} UTC.`);

            const articles = await db.query.articles.findMany({
                where: and(
                    eq($articles.processing_status, 'AI_Processed'),
                    gte($articles.geminiProcessedAt, thirtySixHoursAgo),
                    lt($articles.geminiProcessedAt, now)
                ),
                orderBy: desc($articles.relevance),
                limit: 50,
            });
            totalArticles = articles.length;

            const uniqueSourceIds = [...new Set(articles.map(a => a.sourceId))];
            totalSources = uniqueSourceIds.length;

            console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Found ${totalArticles} AI_Processed articles for brief generation.`);

            if (totalArticles === 0) {
                briefContent = "No sufficiently processed articles from yesterday to generate a brief.";
                briefStatus = "Skipped";
                console.warn(`[DailyBriefGenerator] WARN: Run ID ${runId}: ${briefContent}`);
            } else {
                const rawStories = await processArticlesIntoStories(articles, env, runId);
                usedArticles = rawStories.length;

                const enrichedStories = await analyzeAndEnrichStories(rawStories, env, runId);

                enrichedStories.sort((a, b) => {
                    const importanceOrder = { "Critical": 4, "High": 3, "Moderate": 2, "Low": 1 };
                    return (importanceOrder[b.significance?.assessment] || 0) - (importanceOrder[a.significance?.assessment] || 0);
                });

                const previousDayContext = await fetchLastReportContext(env, getYesterdayDate().toISOString().split('T')[0]);

                briefContent = await generateFinalBrief(enrichedStories, previousDayContext, env, runId);

                briefTitle = await generateBriefTitle(briefContent, env, runId);

                briefTldr = await generateBriefTldr(briefContent, env, runId);

                const usedArticleIds = new Set();
                const usedSourceDomains = new Set();
                enrichedStories.forEach(story => {
                    const storyArticleIds = story.articles_ids || []; 
                    storyArticleIds.forEach(articleId => usedArticleIds.add(articleId));

                    if (story.keySources && story.keySources.provided_articles_sources) {
                        story.keySources.provided_articles_sources.forEach(source => {
                            source.articles.forEach(articleId => {
                                usedArticleIds.add(articleId);
                                const article = articles.find(a => a.id === articleId);
                                if (article && article.url) {
                                    try {
                                        const domain = new URL(article.url).hostname;
                                        usedSourceDomains.add(domain);
                                    } catch (e) { /* ignore invalid URLs */ }
                                }
                            });
                        });
                    }
                });
                usedArticles = usedArticleIds.size;
                usedSources = usedSourceDomains.size;
            }

            console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Saving brief for date ${currentBriefDate} with status ${briefStatus}.`);
            
            await db.insert($dailyBriefs).values({
                briefDate: currentBriefDate,
                title: briefTitle,
                content: briefContent,
                tldr: briefTldr,
                run_id: runId,
                totalArticles: totalArticles,
                totalSources: totalSources,
                usedArticles: usedArticles,
                usedSources: usedSources,
                modelAuthor: briefModelUsed,
                // clusteringParams: JSON.stringify(best_params) // Best params not available in worker
            })
            .onConflictDoUpdate({
                target: $dailyBriefs.briefDate,
                set: {
                    title: briefTitle,
                    content: briefContent,
                    tldr: briefTldr,
                    run_id: runId,
                    totalArticles: totalArticles,
                    totalSources: totalSources,
                    usedArticles: usedArticles,
                    usedSources: usedSources,
                    modelAuthor: briefModelUsed,
                    createdAt: new Date(),
                },
            });
            console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Daily brief saved successfully for date ${currentBriefDate}.`);

            await publishFinalReport({
                briefDate: currentBriefDate,
                title: briefTitle,
                content: briefContent,
                tldr: briefTldr,
                run_id: runId,
                modelAuthor: briefModelUsed,
                totalArticles: totalArticles,
                totalSources: totalSources,
                usedArticles: usedArticles,
                usedSources: usedSources,
            }, env);


        } catch (error) {
            console.error(`[DailyBriefGenerator] CRITICAL ERROR: Run ID ${runId}: Unhandled exception in scheduled handler: ${error?.message || String(error)}`);
            briefStatus = 'Critical_Failure';
            try {
                 await db.insert($dailyBriefs).values({
                    briefDate: currentBriefDate,
                    title: briefTitle,
                    content: briefContent || `Failed to generate brief. Reason: ${error?.message || String(error)}`,
                    tldr: briefTldr || `Failed to generate brief: ${error?.message || String(error)}`,
                    run_id: runId,
                    totalArticles: totalArticles,
                    totalSources: totalSources,
                    usedArticles: usedArticles,
                    usedSources: usedSources,
                    modelAuthor: briefModelUsed,
                    createdAt: new Date(),
                })
                .onConflictDoUpdate({
                    target: $dailyBriefs.briefDate,
                    set: {
                        title: briefTitle,
                        content: briefContent || `Failed to generate brief. Reason: ${error?.message || String(error)}`,
                        tldr: briefTldr || `Failed to generate brief: ${error?.message || String(error)}`,
                        run_id: runId,
                        createdAt: new Date(),
                    },
                });
                console.error(`[DailyBriefGenerator] DEBUG: Run ID ${runId}: Critical error brief saved to DB.`);
            } catch (dbError) {
                console.error(`[DailyBriefGenerator] ERROR: Run ID ${runId}: Failed to save critical error brief: ${dbUpdateError.message}`);
            }
        }
    },
};