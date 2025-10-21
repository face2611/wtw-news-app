// This is the stable version of index.get.ts that produces the "three commas" screen.
// It calls the working /events endpoint and includes the necessary authorization token.

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;
  const secretKey = config.MERIDIAN_SECRET_KEY;

  if (!apiUrl || !secretKey) {
    throw createError({ statusCode: 500, statusMessage: 'API URL or secret key is not configured.' });
  }

  try {
    const responseData = await $fetch(`${apiUrl}/events`, {
      headers: {
        'Authorization': `Bearer ${secretKey}`
      }
    });
    
    return responseData;

  } catch (error) {
    // This is the corrected error logging you found
    console.error('Failed to fetch from /events endpoint:', error);
    
    throw createError({
      statusCode: 502,
      statusMessage: 'Failed to fetch data from the backend worker.',
    });
  }
});