export function useSEO(opts: { title: string; description: string; ogImage: string; ogUrl: string }) {
  return useSeoMeta({
    title: opts.title,
    description: opts.description,
    ogTitle: opts.title,
    ogDescription: opts.description,
    twitterTitle: opts.title,
    twitterDescription: opts.description,
    ogImage: opts.ogImage,
    twitterImage: opts.ogImage,
    twitterCard: 'summary_large_image',
    ogLocale: 'en_US',
    ogUrl: opts.ogUrl,
  });
}
