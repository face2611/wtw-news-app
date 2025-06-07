<script lang="ts" setup>
import { SunIcon, MoonIcon } from '@heroicons/vue/20/solid';
</script>

<template>
  <div>
    <div class="max-w-3xl mx-auto px-6 w-full flex-1">
      <nav class="py-4">
        <ul class="flex space-x-4 items-center font-medium">
          <div class="flex w-full flex-row gap-4">
            <li>
              <NuxtLink class="hover:underline" active-class="underline" to="/"> home </NuxtLink>
            </li>
            <li>
              <NuxtLink class="hover:underline" active-class="underline" to="/briefs"> briefs </NuxtLink>
            </li>
          </div>
          <ClientOnly>
            <button
              class="hover:cursor-pointer"
              v-if="$colorMode.value === 'dark'"
              @click="$colorMode.preference = 'light'"
              aria-label="Switch to light mode"
            >
              <SunIcon class="w-5 h-5 text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-600" />
            </button>
            <button
              class="hover:cursor-pointer"
              v-if="$colorMode.value === 'light'"
              @click="$colorMode.preference = 'dark'"
              aria-label="Switch to dark mode"
            >
              <MoonIcon class="w-5 h-5 text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-600" />
            </button>
          </ClientOnly>
        </ul>
      </nav>
      <div class="h-px w-full bg-gray-300 mb-4" />
      <NuxtPage />
    </div>
    <footer>
      <div class="max-w-3xl mx-auto px-6 py-4">
        <div class="h-px w-full bg-gray-300 mb-4" />
        <p class="text-sm text-center">
          adapted by
          <strong class="underline"><NuxtLink :to="`https://iliane.xyz`">dabsterman</NuxtLink></strong>
          ·
          <span
            >open source on
            <strong class="underline"
              ><NuxtLink to="https://github.com/iliane5/meridian" target="_blank" rel="noopener noreferrer"
                >github</NuxtLink
              ></strong
            ></span
          >
        </p>
      </div>
    </footer>
  </div>
</template>
