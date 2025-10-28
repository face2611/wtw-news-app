// C:\Users\phili\meridian\apps\frontend\src\server/api/briefs/index.get.ts
// --- FINAL FIX: Targets the correct /briefs/last-report endpoint without Authorization header ---

import { defineEventHandler, createError } from 'h3'; // Correct h3 imports
import { useRuntimeConfig } from '#imports'; // CRITICAL FIX: Correct Nuxt import path

export default defineEventHandler(async () => {
  // @ts-ignore // Necessary due to persistent local type resolution issues
  const config = useRuntimeConfig();
  
  const apiUrl = config.public.WORKER_API; 

  // CRITICAL FIX: Ensure explicit check and early exit for API URL
  if (!apiUrl || typeof apiUrl !== 'string') {
    throw createError({ statusCode: 500, statusMessage: 'API URL is not configured.' });
  }

  // Target the correct /briefs/last-report endpoint
  const url = `${apiUrl}/briefs/last-report`; 

  try {
    const responseData = await $fetch(url, {
      method: 'GET',
      // The Authorization header is removed as it's not needed for a public GET
    });
    
    // The response is expected to be { report: ... }
    return responseData;

  } catch (error) {
    console.error('Failed to fetch from /briefs/last-report endpoint:', error);
    
    const e = error as any; 
    const errorBody = e.data || e.message || 'Unknown error';

    throw createError({
      statusCode: 502,
      statusMessage: `Failed to fetch data from the backend worker. Detail: ${errorBody}`,
    });
  }
});