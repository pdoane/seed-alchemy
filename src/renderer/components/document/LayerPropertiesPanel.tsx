import { Box, Flex, Select, Slider, Text, TextField } from "@radix-ui/themes";
import type { Layer } from "../../../shared/types/document";
import type { BlendMode } from "../../../shared/types/image";

const BLEND_MODE_OPTIONS: { value: BlendMode; label: string }[] = [
  { value: "normal", label: "Normal" },
  { value: "multiply", label: "Multiply" },
  { value: "screen", label: "Screen" },
  { value: "overlay", label: "Overlay" },
  { value: "darken", label: "Darken" },
  { value: "lighten", label: "Lighten" },
  { value: "color-dodge", label: "Color Dodge" },
  { value: "color-burn", label: "Color Burn" },
  { value: "hard-light", label: "Hard Light" },
  { value: "soft-light", label: "Soft Light" },
  { value: "difference", label: "Difference" },
  { value: "exclusion", label: "Exclusion" },
];

interface LayerPropertiesPanelProps {
  layer: Layer;
  onOpacityChange: (opacity: number) => void;
  onBlendModeChange: (blendMode: BlendMode) => void;
  onNameChange: (name: string) => void;
  onPositionChange: (x: number, y: number) => void;
}

export function LayerPropertiesPanel({
  layer,
  onOpacityChange,
  onBlendModeChange,
  onNameChange,
  onPositionChange,
}: LayerPropertiesPanelProps) {
  return (
    <Box className="border-t border-[var(--gray-6)] bg-[var(--gray-2)]" p="3">
      <Flex direction="column" gap="3">
        {/* Layer name */}
        <Flex align="center" gap="2">
          <Text size="1" color="gray" style={{ width: 60 }}>
            Name
          </Text>
          <TextField.Root
            size="1"
            value={layer.name}
            onChange={(e) => onNameChange(e.target.value)}
            style={{ flex: 1 }}
          />
        </Flex>

        {/* Opacity */}
        <Flex align="center" gap="2">
          <Text size="1" color="gray" style={{ width: 60 }}>
            Opacity
          </Text>
          <Slider
            size="1"
            value={[layer.opacity]}
            onValueChange={(value) => {
              if (value[0] !== undefined) {
                onOpacityChange(value[0]);
              }
            }}
            min={0}
            max={100}
            step={1}
            style={{ flex: 1 }}
          />
          <Text size="1" style={{ width: 32, textAlign: "right" }}>
            {layer.opacity}%
          </Text>
        </Flex>

        {/* Blend Mode */}
        <Flex align="center" gap="2">
          <Text size="1" color="gray" style={{ width: 60 }}>
            Blend
          </Text>
          <Select.Root
            size="1"
            value={layer.blendMode}
            onValueChange={(value) => onBlendModeChange(value as BlendMode)}
          >
            <Select.Trigger style={{ flex: 1 }} />
            <Select.Content>
              {BLEND_MODE_OPTIONS.map((option) => (
                <Select.Item key={option.value} value={option.value}>
                  {option.label}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
        </Flex>

        {/* Position */}
        <Flex align="center" gap="2">
          <Text size="1" color="gray" style={{ width: 60 }}>
            Position
          </Text>
          <Flex align="center" gap="1" style={{ flex: 1 }}>
            <Text size="1" color="gray">
              X
            </Text>
            <TextField.Root
              size="1"
              type="number"
              value={layer.position.x}
              onChange={(e) =>
                onPositionChange(
                  parseInt(e.target.value) || 0,
                  layer.position.y
                )
              }
              style={{ width: 60 }}
            />
            <Text size="1" color="gray" className="ml-2">
              Y
            </Text>
            <TextField.Root
              size="1"
              type="number"
              value={layer.position.y}
              onChange={(e) =>
                onPositionChange(
                  layer.position.x,
                  parseInt(e.target.value) || 0
                )
              }
              style={{ width: 60 }}
            />
          </Flex>
        </Flex>
      </Flex>
    </Box>
  );
}
