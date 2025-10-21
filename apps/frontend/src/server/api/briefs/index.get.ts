// This is the new, correct code for apps/frontend/src/server/api/briefs/index.get.ts

export default defineEventHandler(async (event) => {
  // Get the runtime configuration we defined in nuxt.config.ts
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;

  if (!apiUrl) {
    throw new Error('WORKER_API is not configured in runtimeConfig.');
  }

  try {
    // Fetch the events/briefs from our actual worker API
    // NOTE: The endpoint in the worker that lists reports is called '/events'
    const briefs = await $fetch(`${apiUrl}/events`, {
      headers: {
        // If your /events endpoint requires an auth token, add it here.
        // Assuming it's a public endpoint for now.
      }
    });
    
    return briefs;

  } catch (error) {
    console.error('Failed to fetch briefs from worker API:', error);
    // Throw an error to let Nuxt know the fetch failed
    throw createError({
      statusCode: 502, // Bad Gateway, indicates an upstream API error
      statusMessage: 'Failed to fetch briefs from the backend worker.',
    });
  }
});