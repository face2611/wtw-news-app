// This is the complete and corrected code for apps/frontend/src/server/api/briefs/index.get.ts

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;
  
  if (!apiUrl) {
    throw createError({ statusCode: 500, statusMessage: 'API URL is not configured.' });
  }

  try {
    // 1. Point to the correct /briefs endpoint.
    // 2. Add a unique timestamp to the URL to bypass any caches.
    const cacheBuster = Date.now();
    const responseData = await $fetch(`${apiUrl}/briefs?cb=${cacheBuster}`);
    
    return responseData;

  } catch (error) {
    // 3. This is the fix for the bug you found. We log the whole error object for better diagnostics.
    console.error('Failed to fetch from /briefs endpoint:', error);
    
    throw createError({
      statusCode: 502,
      statusMessage: 'Failed to fetch data from the backend worker.',
    });
  }
});