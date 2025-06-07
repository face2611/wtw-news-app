<script lang="ts" setup>
import type { Brief } from '~/shared/types';

// Constants (Consider moving to config or keeping here if page-specific)
const WORDS_PER_MINUTE = 300;
const HEADER_OFFSET = 80; // Used by TOC composable
const TOC_WIDTH_CLASS = 'w-64'; // Keep for template styling

// Config & Route
const config = useRuntimeConfig();
const route = useRoute();
const slug = route.path.split('/').pop()?.replaceAll('_', '/'); // Keep original logic for slug

if (!slug) {
  throw createError({ statusCode: 404, statusMessage: 'Brief slug not found in path' });
}

// Data Fetching
// Note: useFetch handles async data fetching correctly within <script setup>
const { data: briefRawData, error: briefError } = await useFetch<Brief>(`/api/briefs/${slug}`);

// Error Handling & Data Processing
if (briefError.value) {
  console.error('Error fetching brief:', briefError.value);
  // Consider more specific error mapping based on briefError.value.statusCode if available
  throw createError({ statusCode: briefError.value.statusCode || 500, statusMessage: 'Error fetching brief data.' });
}
if (!briefRawData.value) {
  throw createError({ statusCode: 404, statusMessage: 'Brief not found.' });
}

// Ensure reactivity and process date
const briefData = toRef({
  ...briefRawData.value,
  createdAt: new Date(briefRawData.value.createdAt),
  // Add reading time calculation here, needs the 'content' field
  readingTime: estimateReadingTime(briefRawData.value.content || ''), // Assuming content field exists
});

// Reading Time Estimation Function (Keep here or move to a utils file)
function estimateReadingTime(content: string): number {
  if (!content) return 0;
  const wordCount = content.trim().split(/\s+/).length;
  return Math.ceil(wordCount / WORDS_PER_MINUTE);
}

// --- Template Refs ---
const articleContentRef = ref<HTMLElement | null>(null); // For TOC and reading time content source
const headerRef = ref<HTMLElement | null>(null); // For sticky detection
const mobileTocRef = ref<HTMLElement | null>(null); // Potentially for mobile menu interaction/styling
const mobileMenuOpen = ref(false); // Mobile TOC menu state

// --- Instantiate Composables ---
const { readingProgress, showBackToTop, scrollToTop } = useReadingProgress();
const { tocItems, activeHeadingId, currentSectionName, scrollToSection } = useTableOfContents({
  contentRef: articleContentRef,
  headerOffset: HEADER_OFFSET,
  // selectors: 'h2, h3, u > strong' // Default is usually fine
});
const { isSticky } = useStickyElement(headerRef); // Observe the header element directly

// --- SEO Metadata ---
const formatDate = computed(() => {
  const date = briefData.value?.date; // Assuming 'date' object exists as per original
  return date?.month ? `${date.month.toLowerCase()} ${date.day}, ${date.year}` : 'Unknown Date';
});

// Use useSeoMeta - Ensure data exists and provide fallbacks
useSeoMeta({
  title: `${briefData.value?.title?.toLowerCase() ?? 'Brief'} | meridian`,
  description: `Intelligence brief for ${formatDate.value}`, // Use computed date
  ogTitle: `${briefData.value?.title ?? 'Intelligence Brief'}`,
  ogDescription: `Intelligence brief for ${formatDate.value}`, // Use computed date
  ogImage: briefData.value?.title // Check if title exists before constructing URL
    ? `${config.public.WORKER_API}/openGraph/brief?title=${encodeURIComponent(briefData.value.title)}&date=${encodeURIComponent(briefData.value.createdAt?.getTime() ?? Date.now())}&articles=${briefData.value.usedArticles ?? 0}&sources=${briefData.value.usedSources ?? 0}`
    : '/default-og-image.png', // Fallback OG image
  ogUrl: `https://news.iliane.xyz/briefs/${slug}`, // Ensure base URL is correct
  twitterCard: 'summary_large_image',
});
</script>

<template>
  <!-- Reading progress bar -->
  <div class="fixed top-0 left-0 w-full bg-white dark:bg-gray-900 h-1 z-50">
    <div
      class="h-full bg-black dark:bg-white transition-all duration-150 ease-out"
      :style="{ width: `${readingProgress}%` }"
    />
  </div>

  <!-- Override the default layout constraints for this page only -->
  <!-- This wrapper pushes its content outside the default max-w-3xl container -->
  <div class="mx-[-1.5rem] w-[calc(100%+3rem)] relative">
    <!-- Inner container to reapply the proper padding -->
    <div class="max-w-3xl mx-auto px-6">
      <!-- Main Content Column -->
      <div>
        <!-- Header section that we'll observe -->
        <header ref="headerRef" class="mb-8">
          <h1 class="text-3xl md:text-4xl font-bold mb-3 leading-tight">
            {{ briefData.title }}
          </h1>
          <div class="flex text-sm text-gray-600 dark:text-gray-400 items-center space-x-2">
            <time>{{ formatDate }}</time>
            <span>•</span>
            <p>{{ estimateReadingTime(briefData.content) }} min read</p>
          </div>
        </header>

        <!-- Mobile TOC (Only visible on small screens) -->
        <div class="xl:hidden mb-8" v-if="tocItems.length > 0">
          <div
            ref="mobileTocRef"
            :class="[
              'z-40 transition-all duration-200 ', // lower z-index to stay below reading indicator
              isSticky ? 'fixed top-1 left-0 right-0' : 'relative', // top-1 to show reading indicator
            ]"
          >
            <button
              @click="mobileMenuOpen = !mobileMenuOpen"
              class="w-full flex items-center justify-between px-6 py-3 text-sm font-semibold text-gray-700 dark:text-gray-300 bg-white/95 dark:bg-gray-900/95 backdrop-blur-md transition-colors duration-200"
            >
              <span>{{ currentSectionName }}</span>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                class="w-4 h-4 transform transition-transform duration-300 ease-in-out"
                :class="{ 'rotate-180': mobileMenuOpen }"
              >
                <path
                  fill-rule="evenodd"
                  d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
                  clip-rule="evenodd"
                />
              </svg>
            </button>
            <Transition
              enter-active-class="transition-all duration-300 ease-out"
              leave-active-class="transition-all duration-200 ease-in"
              enter-from-class="opacity-0 -translate-y-4"
              enter-to-class="opacity-100 translate-y-0"
              leave-from-class="opacity-100 translate-y-0"
              leave-to-class="opacity-0 -translate-y-4"
            >
              <div
                v-if="mobileMenuOpen"
                class="absolute left-0 right-0 bg-white dark:bg-gray-900 shadow-lg p-4 max-h-[60vh] overflow-y-auto"
              >
                <BriefTableOfContents
                  :items="tocItems"
                  :active-heading-id="activeHeadingId"
                  @navigate="
                    id => {
                      scrollToSection(id);
                      mobileMenuOpen = false;
                    }
                  "
                  @active-heading-change="activeHeadingId = $event"
                />
              </div>
            </Transition>
          </div>
        </div>

        <!-- Main article content -->
        <article ref="articleContentRef" class="prose w-full max-w-none" v-html="$md.render(briefData.content)" />

        <!-- Stats and Subscription Form -->
        <div class="mt-16 mb-8">
          <div class="h-px w-full bg-gray-300 dark:bg-gray-700 mb-8" />
          <div class="flex flex-col text-center gap-8">
            <!-- Stat divs remain the same -->
            <div class="grid grid-cols-2 gap-y-6 gap-x-12 text-sm">
              <div>
                <p class="text-gray-600 dark:text-gray-400 mb-1">total articles</p>
                <p class="font-bold text-base">{{ briefData.totalArticles || '-' }}</p>
              </div>
              <div>
                <p class="text-gray-600 dark:text-gray-400 mb-1">total sources</p>
                <p class="font-bold text-base">{{ briefData.totalSources || '-' }}</p>
              </div>
              <div>
                <p class="text-gray-600 dark:text-gray-400 mb-1">used articles</p>
                <p class="font-bold text-base">{{ briefData.usedArticles || '-' }}</p>
              </div>
              <div>
                <p class="text-gray-600 dark:text-gray-400 mb-1">used sources</p>
                <p class="font-bold text-base">{{ briefData.usedSources || '-' }}</p>
              </div>
            </div>

            <div class="text-sm">
              <p class="text-gray-600 dark:text-gray-400 mb-1">final brief generated by</p>
              <p class="font-bold text-base">{{ briefData.model_author }}</p>
            </div>

            <!-- Subscription area -->
            <div class="mt-4 pt-8 border-t border-gray-300 dark:border-gray-700">
              <SubscriptionForm />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Desktop TOC - Positioned absolutely to the left of the content -->
    <aside
      v-if="tocItems.length > 0"
      class="hidden xl:block fixed max-h-[calc(100vh-8rem)] overflow-y-auto pr-8"
      :class="[
        TOC_WIDTH_CLASS,
        // Fixed distance from the left edge at larger screens
        'top-24 right-[78%]',
      ]"
    >
      <ClientOnly>
        <BriefTableOfContents
          :items="tocItems"
          :active-heading-id="activeHeadingId"
          @navigate="scrollToSection"
          @active-heading-change="activeHeadingId = $event"
        />
      </ClientOnly>
    </aside>

    <!-- Back to Top Button -->
    <Transition
      enter-active-class="transition-opacity duration-200"
      leave-active-class="transition-opacity duration-200"
      enter-from-class="opacity-0"
      leave-to-class="opacity-0"
    >
      <button
        v-show="showBackToTop"
        @click="scrollToTop()"
        class="fixed bottom-8 z-50 border border-gray-200 dark:border-gray-950 hover:cursor-pointer right-[max(2rem,calc((100%-68rem)/2))] p-2 rounded-full bg-gray-100 dark:bg-gray-900 text-black dark:text-gray-100 shadow-lg hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors duration-200"
        aria-label="Back to top"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path
            fill-rule="evenodd"
            d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z"
            clip-rule="evenodd"
          />
        </svg>
      </button>
    </Transition>
  </div>
</template>

<style>
/* Global smooth scrolling */
html {
  scroll-behavior: smooth;
}
</style>

<!-- 
/* Optional: Fine-tune prose dark mode if needed */
.dark .prose {
  /* Example: */
  /* color: theme('colors.slate.300'); */
}
.dark .prose strong {
  /* color: theme('colors.slate.100'); */
}
... etc ... -->
