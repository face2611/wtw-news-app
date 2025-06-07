import tailwindcss from '@tailwindcss/vite';

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  app: {
    head: {
      htmlAttrs: { lang: 'en' },
      link: [
        {
          rel: 'icon',
          type: 'image/png',
          href: '/favicon.ico',
        },
      ],
    },
  },

  modules: ['@nuxtjs/color-mode'],

  compatibilityDate: '2025-03-01',
  devtools: {
    enabled: true,
  },
  devServer: {
    host: '0.0.0.0',
  },

  css: ['~/assets/css/main.css'],
  vite: {
    plugins: [tailwindcss()],
  },

  runtimeConfig: {
    public: {
      WORKER_API: 'http://localhost:8787',
    },
    DATABASE_URL: '',
    mailerlite: {
      api_key: '',
      group_id: '',
    },
  },

  srcDir: 'src',

  nitro: {
    prerender: {
      autoSubfolderIndex: false,
    },
  },

  routeRules: {
    // Cache the list of briefs for 1 hour on CDN, 15 mins in browser
    // Allow serving stale data for up to a day while revalidating
    '/api/briefs': {
      cache: {
        maxAge: 60 * 15, // 15 minutes browser cache
        staleMaxAge: 60 * 60 * 24, // 1 day stale-while-revalidate on CDN
      },
    },
    // Cache individual briefs for longer (assuming they don't change once published)
    // Cache for 1 day on CDN, 1 hour in browser
    '/api/briefs/**': {
      // Matches /api/briefs/some-slug, /api/briefs/another-slug etc.
      cache: {
        maxAge: 60 * 60, // 1 hour browser cache
        staleMaxAge: 60 * 60 * 24 * 7, // 1 week stale-while-revalidate on CDN
      },
    },
  },

  colorMode: {
    classSuffix: '',
    preference: 'system',
    fallback: 'system',
  },
});
