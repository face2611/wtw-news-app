<script setup lang="ts">
import SubscriptionForm from '../components/SubscriptionForm.vue';

const { $md } = useNuxtApp();
const config = useRuntimeConfig();

const text = ref(
  `
## Watching The World

Your personal intelligence agency. AI scours the web to compile a daily brief of what is important news, with *actual analysis* beyond headlines.

Distilling the "what", "why", and "so what" behind global events in a succinct format.

### AI-powered daily intelligence brief
- scrapes hundreds of news sources
- clusters related articles
- separates facts from disputed claims
- spots bias patterns
- runs targeted web searches to find context and fill knowledge gaps
- identifies what actually matters
- delivers one clean & engaging brief

### Cut through the noise

Stay decisively informed with our AI-powered daily intelligence brief. 
Each day, our system scours global news, identifying the most critical developments and thematically grouping related stories. 
This provides a structured, prioritized overview, ensuring you quickly grasp impactful events and emerging trends across key sectors like geopolitics, economics, and technology.
`.trim()
);

useSEO({
  title: 'home | WTW',
  description:
    'Gain a strategic edge with our daily AI-generated intelligence brief. Designed for clarity and impact, it transforms a deluge of global news into a concise, structured outline',
  ogImage: `${config.public.WORKER_API}/openGraph/default`,
  ogUrl: `https://news.iliane.xyz/`,
});
</script>

<template>
  <div class="flex flex-col gap-6">
    <div class="prose" v-html="$md.render(text)" />

    <!-- Subscription area -->
    <div class="mt-4 pt-8 pb-6 border-t border-gray-300">
      <SubscriptionForm />
    </div>
  </div>
</template>
