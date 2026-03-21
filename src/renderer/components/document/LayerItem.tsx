import { Box, Flex, IconButton, Text } from "@radix-ui/themes";
import { EyeIcon, EyeSlashIcon, ImageSquareIcon } from "@phosphor-icons/react";
import type { Layer, Asset } from "../../../shared/types/document";

interface LayerItemProps {
  layer: Layer;
  asset?: Asset;
  selected: boolean;
  onSelect: () => void;
  onToggleVisibility: () => void;
  thumbnailUrl?: string;
}

export function LayerItem({
  layer,
  asset,
  selected,
  onSelect,
  onToggleVisibility,
  thumbnailUrl,
}: LayerItemProps) {
  return (
    <Flex
      align="center"
      gap="2"
      px="2"
      py="1"
      className={`cursor-pointer rounded transition-colors ${
        selected ? "bg-[var(--violet-4)]" : "hover:bg-[var(--gray-4)]"
      }`}
      onClick={onSelect}
    >
      <IconButton
        size="1"
        variant="ghost"
        color="gray"
        onClick={(e) => {
          e.stopPropagation();
          onToggleVisibility();
        }}
        aria-label={layer.visible ? "Hide layer" : "Show layer"}
      >
        {layer.visible ? (
          <EyeIcon size={14} weight="fill" />
        ) : (
          <EyeSlashIcon size={14} weight="fill" />
        )}
      </IconButton>

      <Box
        className="flex h-8 w-8 items-center justify-center overflow-hidden rounded bg-[var(--gray-5)]"
        title={asset ? `Asset: ${asset.filename}` : "Empty layer"}
      >
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt=""
            className="h-full w-full object-cover"
          />
        ) : (
          <ImageSquareIcon
            size={16}
            weight="fill"
            className="text-[var(--gray-8)]"
          />
        )}
      </Box>

      <Flex direction="column" className="min-w-0 flex-1">
        <Text size="1" className="truncate">
          {layer.name}
        </Text>
        {layer.opacity < 100 && (
          <Text size="1" color="gray" className="truncate">
            {layer.opacity}%
          </Text>
        )}
      </Flex>
    </Flex>
  );
}
