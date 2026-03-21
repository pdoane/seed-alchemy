import { Box } from "@radix-ui/themes";
import { XIcon } from "@phosphor-icons/react";
import { forwardRef, useState, memo, useCallback, MouseEvent } from "react";
import { api } from "../api/client";
import { useImageStore, type ThumbnailContext } from "../store/imageStore";

interface ImageThumbnailProps {
  filename: string;
  selected?: boolean;
  onClick?: () => void;
  // Stable callback that receives filename - preferred over onClick for performance
  onSelect?: (filename: string) => void;
  // Context for the thumbnail (determines available menu items)
  context?: ThumbnailContext;
  // Reference image specific props
  referenceIndex?: number;
  referenceCount?: number;
  onRemove?: () => void;
  size?: "default" | "small";
  // When false, disables hover effects and selection styling
  selectable?: boolean;
}

export const ImageThumbnail = memo(
  forwardRef<HTMLDivElement, ImageThumbnailProps>(function ImageThumbnail(
    {
      filename,
      selected,
      onClick,
      onSelect,
      context = "gallery",
      referenceIndex,
      referenceCount,
      onRemove,
      size = "default",
      selectable = true,
    },
    ref
  ) {
    const openContextMenu = useImageStore((s) => s.openContextMenu);
    const [isHovered, setIsHovered] = useState(false);

    const handleClick = useCallback(() => {
      if (onClick) {
        onClick();
      } else if (onSelect) {
        onSelect(filename);
      }
    }, [onClick, onSelect, filename]);

    const handleContextMenu = useCallback(
      (e: MouseEvent) => {
        e.preventDefault();
        openContextMenu({
          filename,
          context,
          position: { x: e.clientX, y: e.clientY },
          referenceIndex,
          referenceCount,
          onRemove,
        });
      },
      [
        filename,
        context,
        referenceIndex,
        referenceCount,
        onRemove,
        openContextMenu,
      ]
    );

    const sizeClass = size === "small" ? "h-14 w-14" : "";

    const getHoverClass = () => {
      if (!selectable) return "";
      if (selected) return "ring-2 ring-[var(--violet-9)]";
      if (context === "reference") return "";
      return "hover:ring-2 hover:ring-[var(--gray-8)]";
    };

    return (
      <Box
        ref={ref}
        className={`relative aspect-square overflow-hidden rounded bg-[var(--gray-3)] transition-all duration-150 ${sizeClass} ${
          selectable ? "cursor-pointer" : ""
        } ${getHoverClass()}`}
        onClick={selectable ? handleClick : undefined}
        onContextMenu={handleContextMenu}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <img
          src={api.getImageUrl(filename)}
          alt={`Generated image ${filename}`}
          className="h-full w-full object-contain"
        />
        {onRemove && isHovered && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            className="absolute right-0.5 top-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-black/60 text-white transition-colors hover:bg-red-600"
          >
            <XIcon size={10} weight="bold" />
          </button>
        )}
      </Box>
    );
  })
);
