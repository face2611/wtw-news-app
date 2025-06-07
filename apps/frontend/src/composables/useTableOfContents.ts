export interface TocItem {
  id: string;
  text: string;
  level: number;
}

export interface UseTableOfContentsOptions {
  contentRef: Ref<HTMLElement | null>;
  headerOffset?: number;
  selectors?: string; // e.g., 'h2, h3, u > strong'
}

const DEFAULT_HEADER_OFFSET = 80;
const DEFAULT_SELECTORS = 'h2, h3, u > strong';

// Simple slugify, might need refinement depending on edge cases
const generateSlug = (text: string): string => {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Remove special chars except space/hyphen
    .trim()
    .replace(/\s+/g, '-') // Replace spaces with hyphens
    .slice(0, 50); // Limit length
};

export function useTableOfContents({
  contentRef,
  headerOffset = DEFAULT_HEADER_OFFSET,
  selectors = DEFAULT_SELECTORS,
}: UseTableOfContentsOptions) {
  const tocItems = ref<TocItem[]>([]);
  const activeHeadingId = ref<string | null>(null);
  const mobileMenuOpen = ref(false); // Keep mobile state here if tied to TOC display

  let observer: IntersectionObserver | null = null;

  const generateToc = () => {
    if (!contentRef.value) return;

    const elements = contentRef.value.querySelectorAll(selectors);
    const newTocItems: TocItem[] = [];
    const observedElements: Element[] = []; // Keep track of elements to observe

    elements.forEach((el, index) => {
      let level: number;
      let text = el.textContent?.trim() || '';
      let targetElement: HTMLElement = el as HTMLElement;

      if (el.tagName === 'H2') level = 2;
      else if (el.tagName === 'H3') level = 3;
      else if (el.tagName === 'STRONG' && el.parentElement?.tagName === 'U') {
        level = 5; // Special level for topics
        targetElement = el.parentElement; // Target the <u> tag
      } else {
        return; // Skip unrecognized elements
      }

      // Ensure unique ID even if slug is identical
      const id = `${level === 5 ? 'topic' : 'section'}-${index}-${generateSlug(text)}`;

      if (text && targetElement) {
        targetElement.id = id; // Assign ID
        newTocItems.push({ id, text, level });
        observedElements.push(targetElement); // Add element for intersection observer
      }
    });

    tocItems.value = newTocItems;
    setupIntersectionObserver(observedElements); // Setup observer after generating TOC
  };

  const setupIntersectionObserver = (elements: Element[]) => {
    // Disconnect previous observer if exists
    if (observer) {
      observer.disconnect();
    }

    // Observer options: trigger when heading is near the top of the viewport
    const options = {
      rootMargin: `-${headerOffset - 1}px 0px -${window.innerHeight - headerOffset - 50}px 0px`, // Adjust bottom margin as needed
      threshold: 0, // Trigger as soon as any part enters/leaves the rootMargin
    };

    observer = new IntersectionObserver(entries => {
      // Find the topmost visible entry
      let topmostVisibleEntry: IntersectionObserverEntry | null = null;
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Prioritize the entry closest to the top boundary defined by rootMargin
          if (!topmostVisibleEntry || entry.boundingClientRect.top < topmostVisibleEntry.boundingClientRect.top) {
            topmostVisibleEntry = entry;
          }
        }
      });

      if (topmostVisibleEntry) {
        activeHeadingId.value = (topmostVisibleEntry as IntersectionObserverEntry).target.id;
      } else {
        // If no entry is intersecting within the top margin, check if we scrolled past the first item
        if (tocItems.value.length > 0 && window.scrollY > document.getElementById(tocItems.value[0].id)!.offsetTop) {
          // Potentially keep the last active ID, or find the last item scrolled past
          // For simplicity, let's just keep the *last* one that *was* active if nothing is currently in the top zone
          // activeHeadingId.value remains unchanged unless explicitly cleared or updated
        } else {
          // Scrolled to the very top above the first item
          activeHeadingId.value = null;
        }
      }
    }, options);

    elements.forEach(el => observer!.observe(el));
  };

  // Computed property for the "current section name" shown in mobile/dropdown
  const currentSectionName = computed(() => {
    if (!activeHeadingId.value) {
      return 'on this page'; // Default text
    }
    const activeItem = tocItems.value.find(item => item.id === activeHeadingId.value);
    // Maybe find the parent H2 if the active item is H3/topic? Depends on desired UX.
    // For now, just use the active item's text.
    return activeItem ? activeItem.text.toLowerCase() : 'on this page';
  });

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      const elementPosition = el.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
      window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
      mobileMenuOpen.value = false; // Close mobile menu on selection
    }
  };

  onMounted(() => {
    // Ensure DOM is ready before querying elements
    nextTick(() => {
      generateToc();
    });
  });

  onUnmounted(() => {
    if (observer) {
      observer.disconnect();
    }
  });

  // Optional: Watch for content changes if the article content could be dynamic
  // watch(contentRef, () => { nextTick(generateToc); });

  return {
    tocItems,
    activeHeadingId,
    currentSectionName,
    mobileMenuOpen,
    generateToc, // Expose if manual regeneration is needed
    scrollToSection,
  };
}
