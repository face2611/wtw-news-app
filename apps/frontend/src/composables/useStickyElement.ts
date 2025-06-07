export function useStickyElement(
  targetRef: Ref<HTMLElement | null>,
  options: IntersectionObserverInit = { threshold: 0 }
) {
  const isSticky = ref(false);
  let observer: IntersectionObserver | null = null;

  const setupObserver = () => {
    if (observer) observer.disconnect(); // Clean up previous observer

    if (targetRef.value) {
      // Use a placeholder element *before* the target to detect when the target *would* leave the screen top
      // Or observe the target itself and check entry.boundingClientRect.top <= options.rootMargin top value (if set)
      // Simpler approach: Observe the target and set sticky when it's *not* intersecting the *top* of the viewport.
      // Let's observe a sentinel element placed *above* the header for simplicity if possible,
      // otherwise observe the header itself and use rootMargin.

      // Assuming we observe the element *itself* and want it sticky when it scrolls *off* the top.
      // We need a negative top margin equal to the element's height or just 1px if we only care when it *starts* scrolling off.
      // Let's stick to the original logic: observe the element, become sticky when *not* intersecting.
      // This requires the element to *start* within the viewport.

      observer = new IntersectionObserver(
        ([entry]) => {
          // Becomes sticky when the *observed element* is no longer intersecting the viewport (at the top)
          // This interpretation might depend on where the observed element is relative to the sticky element itself.
          // If targetRef *is* the element becoming sticky, this works.
          isSticky.value = !entry.isIntersecting;
        },
        options // Use provided options (e.g., { threshold: 0 })
      );
      observer.observe(targetRef.value);
    }
  };

  onMounted(() => {
    // Need to wait for the element to be mounted and potentially rendered
    nextTick(setupObserver);
  });

  onUnmounted(() => {
    if (observer) {
      observer.disconnect();
    }
  });

  // Re-setup if the target element changes (e.g., v-if)
  watch(targetRef, () => {
    nextTick(setupObserver);
  });

  return {
    isSticky,
  };
}
