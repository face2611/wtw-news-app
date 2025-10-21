// This is the FINAL, CORRECTED code for apps/frontend/src/server/api/briefs/index.get.ts

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;
  const secretKey = config.MERIDIAN_SECRET_KEY; // <-- Get the secret key

  if (!apiUrl || !secretKey) {
    throw createError({ statusCode: 500, statusMessage: 'API URL or secret key is not configured.' });
  }

  try {
    const briefs = await $fetch(`${apiUrl}/briefs`, { 
      headers: {
        // THIS IS THE CRITICAL MISSING PIECE
        'Authorization': `Bearer ${secretKey}`
      }
    });
    
    return briefs;

  } catch (error) {
    console.error('Failed to fetch briefs from worker API:', error);
    throw createError({
      statusCode: 502,
      statusMessage: 'Failed to fetch briefs from the backend worker.',
    });
  }
});