import { useEffect, useRef, useState, useCallback } from "react";
import {
  Stage,
  Layer as KonvaLayer,
  Rect,
  Image as KonvaImage,
} from "react-konva";
import { Box, Flex, Text } from "@radix-ui/themes";
import { ImageSquareIcon } from "@phosphor-icons/react";
import type {
  ToolType,
  Layer,
  LayerGroup,
  Asset,
  CanvasSize,
} from "../../../shared/types/document";

interface DocumentViewportProps {
  activeTool: ToolType;
  canvasSize?: CanvasSize;
  layers?: (Layer | LayerGroup)[];
  assets?: Asset[];
  getAssetUrl?: (filename: string) => string;
  selectedLayerId?: string | null;
  onLayerSelect?: (layerId: string) => void;
  onAssetDrop?: (assetId: string, x: number, y: number) => void;
}

// Component for rendering a single layer image
function LayerImage({
  layer,
  imageUrl,
  isSelected,
  onSelect,
}: {
  layer: Layer;
  imageUrl?: string;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!imageUrl) {
      setImage(null);
      return;
    }

    const img = new window.Image();
    img.crossOrigin = "use-credentials";
    img.src = imageUrl;
    img.onload = () => setImage(img);
    img.onerror = () => setImage(null);

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  if (!layer.visible || !image) return null;

  return (
    <KonvaImage
      image={image}
      x={layer.position.x}
      y={layer.position.y}
      opacity={layer.opacity / 100}
      onClick={onSelect}
      onTap={onSelect}
      // Show selection border if selected
      stroke={isSelected ? "var(--violet-9)" : undefined}
      strokeWidth={isSelected ? 2 : 0}
    />
  );
}

export function DocumentViewport({
  activeTool,
  canvasSize,
  layers = [],
  assets = [],
  getAssetUrl,
  selectedLayerId,
  onLayerSelect,
  onAssetDrop,
}: DocumentViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({
    width: 800,
    height: 600,
  });
  const [isDragOver, setIsDragOver] = useState(false);

  // Update container size on resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      window.removeEventListener("resize", updateSize);
      resizeObserver.disconnect();
    };
  }, []);

  // Calculate scale to fit canvas in viewport with padding
  const getViewTransform = useCallback(() => {
    if (!canvasSize) return { scale: 1, offsetX: 0, offsetY: 0 };

    const padding = 40;
    const availableWidth = containerSize.width - padding * 2;
    const availableHeight = containerSize.height - padding * 2;

    const scaleX = availableWidth / canvasSize.width;
    const scaleY = availableHeight / canvasSize.height;
    const scale = Math.min(scaleX, scaleY, 1); // Don't scale up beyond 1:1

    const scaledWidth = canvasSize.width * scale;
    const scaledHeight = canvasSize.height * scale;
    const offsetX = (containerSize.width - scaledWidth) / 2;
    const offsetY = (containerSize.height - scaledHeight) / 2;

    return { scale, offsetX, offsetY };
  }, [canvasSize, containerSize]);

  // Helper to find asset by filename
  const findAsset = (filename?: string) =>
    filename ? assets.find((a) => a.filename === filename) : undefined;

  // Cursor style based on active tool
  const getCursorClass = () => {
    switch (activeTool) {
      case "brush":
      case "eraser":
        return "cursor-crosshair";
      case "select":
        return "cursor-default";
      case "rectangle":
      case "lasso":
      case "wand":
        return "cursor-crosshair";
      default:
        return "cursor-default";
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes("application/x-asset-id")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      setIsDragOver(true);
    }
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const assetId = e.dataTransfer.getData("application/x-asset-id");
    if (!assetId || !onAssetDrop || !canvasSize) return;

    // Calculate drop position in canvas coordinates
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const { scale, offsetX, offsetY } = getViewTransform();
    const dropX = (e.clientX - rect.left - offsetX) / scale;
    const dropY = (e.clientY - rect.top - offsetY) / scale;

    // Clamp to canvas bounds
    const x = Math.max(0, Math.min(canvasSize.width, dropX));
    const y = Math.max(0, Math.min(canvasSize.height, dropY));

    onAssetDrop(assetId, Math.round(x), Math.round(y));
  };

  // No document selected state
  if (!canvasSize) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        className={`h-full bg-[var(--gray-3)] ${getCursorClass()}`}
      >
        <Box className="flex h-64 w-64 items-center justify-center rounded-lg border-2 border-dashed border-[var(--gray-6)] bg-[var(--gray-4)]">
          <Flex direction="column" align="center" gap="2">
            <ImageSquareIcon
              size={48}
              weight="thin"
              className="text-[var(--gray-8)]"
            />
            <Text size="2" color="gray">
              No document selected
            </Text>
          </Flex>
        </Box>
      </Flex>
    );
  }

  const { scale, offsetX, offsetY } = getViewTransform();

  return (
    <div
      ref={containerRef}
      className={`h-full w-full bg-[var(--gray-3)] ${getCursorClass()} ${isDragOver ? "ring-2 ring-inset ring-[var(--violet-8)]" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Stage width={containerSize.width} height={containerSize.height}>
        <KonvaLayer>
          {/* Canvas background (checkerboard for transparency) */}
          <Rect
            x={offsetX}
            y={offsetY}
            width={canvasSize.width * scale}
            height={canvasSize.height * scale}
            fill="#1a1a1a"
            shadowColor="black"
            shadowBlur={10}
            shadowOpacity={0.3}
            shadowOffsetX={2}
            shadowOffsetY={2}
          />
        </KonvaLayer>

        {/* Layer content - render in order (bottom to top) */}
        <KonvaLayer
          x={offsetX}
          y={offsetY}
          scaleX={scale}
          scaleY={scale}
          clipX={0}
          clipY={0}
          clipWidth={canvasSize.width}
          clipHeight={canvasSize.height}
        >
          {layers.map((layer) => {
            // Skip layer groups for now
            if ("children" in layer) return null;

            const asset = findAsset(layer.asset);
            const imageUrl =
              asset && getAssetUrl ? getAssetUrl(asset.filename) : undefined;

            return (
              <LayerImage
                key={layer.id}
                layer={layer}
                imageUrl={imageUrl}
                isSelected={layer.id === selectedLayerId}
                onSelect={() => onLayerSelect?.(layer.id)}
              />
            );
          })}
        </KonvaLayer>
      </Stage>
    </div>
  );
}
