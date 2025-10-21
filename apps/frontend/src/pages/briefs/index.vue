<script setup lang="ts">
useSEO({
  title: 'briefs | WatchingTheWorld',
  description: 'list of all briefs',
  ogImage: `${useRuntimeConfig().public.WORKER_API}/og/default`,
  ogUrl: `https://news.iliane.xyz/briefs`,
});

const { data: briefsList, error } = await useFetch('/api/briefs');
if (error.value !== null) {
  console.error('Failed to fetch briefs list');
  throw createError({ statusCode: 500, statusMessage: 'Failed to fetch briefs list' });
}

// Helper function to format the 'YYYY-MM-DD' date string for display
function formatDate(dateString: string) {
  if (!dateString) return '';
  // Appending T00:00:00 ensures the date is parsed in UTC, avoiding timezone issues.
  const date = new Date(dateString + 'T00:00:00');
  const monthName = date.toLocaleString('en-us', { month: 'long', timeZone: 'UTC' });
  const day = date.getUTCDate();
  const year = date.getUTCFullYear();
  return `${monthName.toLowerCase()} ${day}, ${year}`;
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <!-- The template now uses brief.briefDate for the link and calls our new formatDate function for display -->
    <NuxtLink v-for="brief in briefsList" :key="brief.id" class="group" :to="`/briefs/${brief.briefDate}`">
      <p class="text-xl font-bold group-hover:underline">{{ brief.title }}</p>
      <p class="text-sm text-gray-600 mt-1">
        {{ formatDate(brief.briefDate) }}
      </p>
    </NuxtLink>
  </div>
</template>