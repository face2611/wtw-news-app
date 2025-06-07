import { WorkflowStep } from 'cloudflare:workers';

type RateLimiterOptions = {
  maxConcurrent: number;
  globalCooldownMs: number;
  domainCooldownMs: number;
};

type BatchItem<IdType = number | string> = {
  id: IdType;
  url: string;
};

export class DomainRateLimiter<T extends BatchItem<I>, I = number | string> {
  private lastDomainAccess = new Map<string, number>();
  private options: RateLimiterOptions;

  constructor(options: RateLimiterOptions) {
    this.options = options;
  }

  async processBatch<R>(
    items: T[],
    step: WorkflowStep,
    processItem: (item: T, domain: string) => Promise<R>
  ): Promise<R[]> {
    const results: R[] = [];
    const remainingItems = [...items];

    while (remainingItems.length > 0) {
      const currentBatch: T[] = [];
      const currentTime = Date.now();

      // Select items for current batch based on domain cooldown
      for (const item of [...remainingItems]) {
        if (currentBatch.length >= this.options.maxConcurrent) break;

        try {
          const domain = new URL(item.url).hostname;
          const lastAccess = this.lastDomainAccess.get(domain) || 0;

          if (currentTime - lastAccess >= this.options.domainCooldownMs) {
            currentBatch.push(item);
            // Remove from remaining items
            const idx = remainingItems.findIndex(i => i.id === item.id);
            if (idx >= 0) remainingItems.splice(idx, 1);
          }
        } catch (e) {
          // Skip invalid URLs
          const idx = remainingItems.findIndex(i => i.id === item.id);
          if (idx >= 0) remainingItems.splice(idx, 1);
        }
      }

      if (currentBatch.length === 0) {
        // Nothing ready yet, wait for next domain to be ready
        const nextReady = Math.min(
          ...Array.from(this.lastDomainAccess.values()).map(
            time => this.options.domainCooldownMs - (currentTime - time)
          )
        );
        await step.sleep(`waiting for domain cooldown (${Math.round(nextReady / 1000)}s)`, Math.max(500, nextReady));
        continue;
      }

      // Process current batch in parallel
      const batchResults = await Promise.allSettled(
        currentBatch.map(async item => {
          try {
            const domain = new URL(item.url).hostname;
            this.lastDomainAccess.set(domain, Date.now());
            return await processItem(item, domain);
          } catch (error) {
            console.error(`Error processing item ${item.id}: ${error}`);
            throw error;
          }
        })
      );

      // Add results
      batchResults.forEach(result => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        }
      });

      // Apply global cooldown between batches if we have more items to process
      if (remainingItems.length > 0) {
        await step.sleep(
          `global rate limit (${Math.round(this.options.globalCooldownMs / 1000)}s)`,
          this.options.globalCooldownMs
        );
      }
    }

    return results;
  }
}
