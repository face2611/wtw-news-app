<script setup lang="ts">
const COOKIE_NAME = 'WTW_subscribed';
const LEGACY_STORAGE_KEY = 'WTW_subscribed';

// Subscription state
const email = ref('');
const isSubmitting = ref(false);
const errorMessage = ref('');
const hasSubscribed = useCookie<string | null>(COOKIE_NAME);

// Migrate old localStorage data if it exists
onMounted(() => {
  const legacyValue = localStorage.getItem(LEGACY_STORAGE_KEY);
  if (legacyValue === 'true' && !hasSubscribed.value) {
    hasSubscribed.value = 'true';
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  }
});

/**
 * Subscription form handlers
 */
const handleSubmit = async () => {
  isSubmitting.value = true;
  errorMessage.value = '';

  try {
    const response = await $fetch('/api/subscribe', {
      method: 'POST',
      body: { email: email.value },
    });

    if (!response.success) {
      throw new Error(response.message || 'Failed to subscribe');
    }

    email.value = '';
    hasSubscribed.value = 'true';
  } catch (error: unknown) {
    errorMessage.value = error instanceof Error ? error.message : 'Something went wrong, please try again.';
    console.error('Subscription error:', error);
  } finally {
    isSubmitting.value = false;
  }
};

const handleChangeEmail = () => {
  hasSubscribed.value = null;
  errorMessage.value = '';
};
</script>

<template>
  <div>
    <div v-if="!hasSubscribed" class="gap-2 text-sm flex flex-col items-center">
      <p>Want this brief in your inbox? Sign up for updates</p>
      <form @submit.prevent="handleSubmit" class="flex flex-col group max-w-md mx-auto">
        <div class="flex border border-zinc-300">
          <input
            v-model="email"
            type="email"
            placeholder="your@email.com"
            required
            :aria-invalid="!!errorMessage"
            aria-describedby="subscription-error"
            class="flex-grow px-4 py-2 focus:outline-zinc-400"
          />
          <button
            type="submit"
            class="bg-zinc-300 text-zinc-700 hover:cursor-pointer px-4 py-2 font-medium hover:bg-zinc-400 dark:hover:bg-zinc-400 transition-colors"
            :disabled="isSubmitting"
          >
            {{ isSubmitting ? 'Sending...' : 'Subscribe' }}
          </button>
        </div>
        <div v-if="errorMessage" id="subscription-error" class="text-red-600 text-xs mt-2" role="alert">
          {{ errorMessage }}
        </div>
      </form>
    </div>
    <div v-else class="text-center text-sm">
      <p>You're subscribed to our updates!</p>
      <button @click="handleChangeEmail" class="underline mt-2 text-xs">Change email</button>
    </div>
  </div>
</template>
