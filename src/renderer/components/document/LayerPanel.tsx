import { Box, Button, Flex, Text } from "@radix-ui/themes";
import { PlusIcon, TrashIcon } from "@phosphor-icons/react";
import { LayerItem } from "./LayerItem";
import { LayerPropertiesPanel } from "./LayerPropertiesPanel";
import type { Layer, LayerGroup, Asset } from "../../../shared/types/document";
import type { BlendMode } from "../../../shared/types/image";

interface LayerPanelProps {
  layers: (Layer | LayerGroup)[];
  assets: Asset[];
  selectedLayerId: string | null;
  onSelectLayer: (layerId: string) => void;
  onToggleVisibility: (layerId: string) => void;
  onAddLayer: () => void;
  onRemoveLayer?: (layerId: string) => void;
  getAssetUrl: (filename: string) => string;
  onLayerOpacityChange?: (layerId: string, opacity: number) => void;
  onLayerBlendModeChange?: (layerId: string, blendMode: BlendMode) => void;
  onLayerNameChange?: (layerId: string, name: string) => void;
  onLayerPositionChange?: (layerId: string, x: number, y: number) => void;
}

export function LayerPanel({
  layers,
  assets,
  selectedLayerId,
  onSelectLayer,
  onToggleVisibility,
  onAddLayer,
  onRemoveLayer,
  getAssetUrl,
  onLayerOpacityChange,
  onLayerBlendModeChange,
  onLayerNameChange,
  onLayerPositionChange,
}: LayerPanelProps) {
  // Display layers in reverse order (top layer first, like Photoshop)
  const reversedLayers = [...layers].reverse();

  // Helper to find asset by filename
  const findAsset = (filename?: string) =>
    filename ? assets.find((a) => a.filename === filename) : undefined;

  // Helper to get thumbnail URL for a layer
  const getThumbnailUrl = (layer: Layer) => {
    const asset = findAsset(layer.asset);
    return asset ? getAssetUrl(asset.filename) : undefined;
  };

  // Find selected layer (only non-group layers for now)
  const selectedLayer = selectedLayerId
    ? (layers.find((l) => !("children" in l) && l.id === selectedLayerId) as
        | Layer
        | undefined)
    : undefined;

  return (
    <Flex direction="column" className="h-full bg-[var(--gray-2)]">
      <Flex
        align="center"
        justify="between"
        px="3"
        py="2"
        className="border-b border-[var(--gray-6)]"
      >
        <Text size="2" weight="medium">
          Layers
        </Text>

        <Flex gap="1">
          <Button
            size="1"
            variant="ghost"
            color="gray"
            onClick={() => selectedLayerId && onRemoveLayer?.(selectedLayerId)}
            disabled={!selectedLayerId}
          >
            <TrashIcon size={14} weight="bold" />
          </Button>
          <Button size="1" variant="ghost" color="gray" onClick={onAddLayer}>
            <PlusIcon size={14} weight="bold" />
          </Button>
        </Flex>
      </Flex>

      <Box className="flex-1 overflow-y-auto" py="1">
        {reversedLayers.map((item) => {
          // Skip layer groups for now (will be handled later)
          if ("children" in item) {
            return null;
          }
          const layer = item as Layer;
          const asset = findAsset(layer.asset);
          return (
            <LayerItem
              key={layer.id}
              layer={layer}
              asset={asset}
              selected={layer.id === selectedLayerId}
              onSelect={() => onSelectLayer(layer.id)}
              onToggleVisibility={() => onToggleVisibility(layer.id)}
              thumbnailUrl={getThumbnailUrl(layer)}
            />
          );
        })}
      </Box>

      {/* Layer properties panel */}
      {selectedLayer && (
        <LayerPropertiesPanel
          layer={selectedLayer}
          onOpacityChange={(opacity) =>
            onLayerOpacityChange?.(selectedLayer.id, opacity)
          }
          onBlendModeChange={(blendMode) =>
            onLayerBlendModeChange?.(selectedLayer.id, blendMode)
          }
          onNameChange={(name) => onLayerNameChange?.(selectedLayer.id, name)}
          onPositionChange={(x, y) =>
            onLayerPositionChange?.(selectedLayer.id, x, y)
          }
        />
      )}
    </Flex>
  );
}
