// C:\Users\phili\meridian\apps\frontend\src\pages\briefs\index.vue (Redirect Logic FIX)

<script setup lang="ts">
import { useFetch, navigateTo, createError } from '#app'; // Ensure all imports are correct

// Fetch the *single* latest report object wrapper: { report: { ... } }
const { data: reportWrapper, error } = await useFetch<{ report: { briefDate: string } | null }>('/api/briefs');

if (error.value !== null) {
  console.error('Failed to fetch latest brief data for redirect:', error.value);
  throw createError({ statusCode: 500, statusMessage: `Failed to find latest report.` });
}

// Extract the actual report object
const latestReport = reportWrapper.value?.report;

if (latestReport && latestReport.briefDate) {
  // Redirect using the confirmed briefDate slug
  await navigateTo(`/briefs/${latestReport.briefDate}`);
} else {
    // If no report found (e.g., empty DB)
    await navigateTo(`/briefs/list`);
}
</script>
<template>
  <div>
    <p>Redirecting to the latest report...</p>
  </div>
</template>