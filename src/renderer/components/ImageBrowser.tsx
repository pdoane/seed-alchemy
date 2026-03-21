import { Box, Flex, ScrollArea, Slider, Text } from "@radix-ui/themes";
import { useEffect, useRef, useState, useCallback } from "react";
import { useImageStore } from "../store/imageStore";
import { ImageThumbnail } from "./ImageThumbnail";

export function ImageBrowser() {
  const images = useImageStore((s) => s.images);
  const selectedFilename = useImageStore((s) => s.ui.selectedFilename);
  const navigateTo = useImageStore((s) => s.navigateTo);
  const setBrowserColumns = useImageStore((s) => s.setBrowserColumns);
  const selectedRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const [thumbnailSize, setThumbnailSize] = useState(80);

  // Calculate column count from the grid's computed style
  const updateColumnCount = useCallback(() => {
    if (gridRef.current) {
      const style = getComputedStyle(gridRef.current);
      const columns = style.gridTemplateColumns.split(" ").length;
      setBrowserColumns(columns);
    }
  }, [setBrowserColumns]);

  // Track grid size changes to update column count
  useEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;

    const observer = new ResizeObserver(() => {
      updateColumnCount();
    });
    observer.observe(grid);
    updateColumnCount();

    return () => observer.disconnect();
  }, [updateColumnCount]);

  // Update columns when thumbnail size changes
  useEffect(() => {
    // Delay to let the grid reflow
    const timer = setTimeout(updateColumnCount, 0);
    return () => clearTimeout(timer);
  }, [thumbnailSize, updateColumnCount]);

  // Scroll selected image into view
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "nearest",
      });
    }
  }, [selectedFilename]);

  return (
    <Flex direction="column" height="100%" className="bg-[var(--gray-2)]">
      <Box px="2" py="2" className="border-b border-[var(--gray-6)]">
        <Slider
          size="1"
          value={[thumbnailSize]}
          onValueChange={(value) => setThumbnailSize(value[0] ?? 80)}
          min={48}
          max={120}
          step={4}
        />
      </Box>
      <ScrollArea scrollbars="vertical" style={{ flex: 1 }}>
        {images.length === 0 ? (
          <Flex align="center" justify="center" width="100%" py="4">
            <Text size="1" color="gray" align="center">
              No images yet
            </Text>
          </Flex>
        ) : (
          <div
            ref={gridRef}
            className="gap-1 p-1"
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(auto-fill, minmax(${thumbnailSize}px, 1fr))`,
            }}
          >
            {images.map((filename) => (
              <ImageThumbnail
                key={filename}
                ref={selectedFilename === filename ? selectedRef : undefined}
                filename={filename}
                selected={selectedFilename === filename}
                onSelect={navigateTo}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </Flex>
  );
}
