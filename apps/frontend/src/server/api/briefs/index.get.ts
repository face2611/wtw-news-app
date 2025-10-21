// Reverting to the version that calls the working /events endpoint.

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;
  const secretKey = config.MERIDIAN_SECRET_KEY;

  if (!apiUrl || !secretKey) {
    throw createError({ statusCode: 500, statusMessage: 'API URL or secret key is not configured.' });
  }

  try {
    // REVERTED: Call the /events endpoint, which we know connects successfully.
    const responseData = await $fetch(`${apiUrl}/events`, {
      headers: {
        'Authorization': `Bearer ${secretKey}`
      }
    });
    
    return responseData;

  } catch (error) {
    console.error('Failed to fetch from /events endpoint:', error.data);
    throw createError({
      statusCode: 502,
      statusMessage: 'Failed to fetch data from the backend worker.',
    });
  }
});