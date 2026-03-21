import { useEffect, useCallback } from "react";
import { useImageStore } from "../store/imageStore";
import { useAppStore } from "../store/appStore";

// Elements that should block arrow key navigation
function isInteractiveElement(element: Element | null): boolean {
  if (!element) return false;
  const tagName = element.tagName.toLowerCase();
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    element.getAttribute("contenteditable") === "true"
  );
}

export function useKeyboardShortcuts() {
  const mode = useAppStore((s) => s.mode);
  const images = useImageStore((s) => s.images);
  const selectedFilename = useImageStore((s) => s.ui.selectedFilename);
  const navigateTo = useImageStore((s) => s.navigateTo);
  const generate = useImageStore((s) => s.generate);
  const isGenerating = useImageStore((s) => s.isGenerating);
  const params = useImageStore((s) => s.params);
  const removeImage = useImageStore((s) => s.removeImage);
  const browserColumns = useImageStore((s) => s.browserColumns);

  const canGenerate = params.model && params.prompt && !isGenerating;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Only handle shortcuts in image mode for now
      if (mode !== "image") return;

      const isMeta = e.metaKey || e.ctrlKey;

      // Cmd+Enter: Generate
      if (isMeta && e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        if (canGenerate) {
          generate();
        }
        return;
      }

      // Cmd+Backspace: Delete selected image
      if (isMeta && e.key === "Backspace") {
        e.preventDefault();
        if (selectedFilename && confirm("Delete this image?")) {
          removeImage(selectedFilename);
        }
        return;
      }

      // Arrow keys: Image navigation (only when not in interactive element)
      if (isInteractiveElement(document.activeElement)) return;

      if (images.length === 0) return;

      const currentIndex = selectedFilename
        ? images.indexOf(selectedFilename)
        : -1;

      if (e.key === "ArrowLeft" && currentIndex > 0) {
        e.preventDefault();
        navigateTo(images[currentIndex - 1] ?? null);
      } else if (e.key === "ArrowRight" && currentIndex < images.length - 1) {
        e.preventDefault();
        const newIndex = currentIndex === -1 ? 0 : currentIndex + 1;
        navigateTo(images[newIndex] ?? null);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const newIndex = currentIndex - browserColumns;
        if (newIndex >= 0) {
          navigateTo(images[newIndex] ?? null);
        }
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        const newIndex =
          currentIndex === -1 ? 0 : currentIndex + browserColumns;
        if (newIndex < images.length) {
          navigateTo(images[newIndex] ?? null);
        }
      }
    },
    [
      mode,
      images,
      selectedFilename,
      navigateTo,
      generate,
      canGenerate,
      removeImage,
      browserColumns,
    ]
  );

  useEffect(() => {
    // Use capture phase so shortcuts run before UI components handle the event
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [handleKeyDown]);
}
