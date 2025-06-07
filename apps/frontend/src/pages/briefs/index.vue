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
</script>

<template>
  <div class="flex flex-col gap-6">
    <NuxtLink v-for="brief in briefsList" :key="brief.id" class="group" :to="`/briefs/${brief.slug}`">
      <p class="text-xl font-bold group-hover:underline">{{ brief.title }}</p>
      <p class="text-sm text-gray-600 mt-1">
        {{ brief.date?.month.toLowerCase() }} {{ brief.date?.day }}, {{ brief.date?.year }}
      </p>
    </NuxtLink>
  </div>
</template>
