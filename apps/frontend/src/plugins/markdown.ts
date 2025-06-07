import MarkdownIt from 'markdown-it';
import mdColorDefault from 'markdown-it-color';
// @ts-ignore - no types for this package
import mdTaskListsDefault from 'markdown-it-deflist';

// Helper to get the actual function, handling CJS/ESM differences
const unwrapDefault = (mod: any) => mod.default || mod;

const markdownItColor = unwrapDefault(mdColorDefault);
const markdownItTaskLists = unwrapDefault(mdTaskListsDefault);

import { defineNuxtPlugin } from '#app';

export default defineNuxtPlugin({
  name: 'markdown-it',
  setup() {
    const md = new MarkdownIt({
      linkify: true,
      breaks: true,
      typographer: true,
      html: true, // Be careful with this if markdown comes from users!
    })
      .use(markdownItTaskLists)
      .use(markdownItColor, { defaultClassName: 'text-primary' });

    return {
      provide: {
        md: md, // Provide the configured instance
      },
    };
  },
});
