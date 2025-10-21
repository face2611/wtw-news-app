// This is the stable version of index.get.ts that produces the "blank screen" without a 500 error.

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const apiUrl = config.public.WORKER_API;
  
  if (!apiUrl) {
    throw createError({ statusCode: 500, statusMessage: 'API URL is not configured.' });
  }

  try {
    const responseData = await $fetch(`${apiUrl}/briefs`);
    return responseData;

  } catch (error) {
    console.error('Failed to fetch from /briefs endpoint:', error);
    throw createError({
      statusCode: 502,
      statusMessage: 'Failed to fetch data from the backend worker.',
    });
  }
});