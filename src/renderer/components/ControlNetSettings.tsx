import {
  Box,
  Flex,
  IconButton,
  Select,
  Slider,
  Text,
  Tooltip,
} from "@radix-ui/themes";
import { PlayIcon, PlusIcon, XIcon } from "@phosphor-icons/react";
import type { Architecture } from "../../shared/types/models";
import type {
  ControlNetConfig,
  ControlNetType,
  PreprocessorInfo,
} from "../../shared/types/controlnet";
import {
  CONTROLNET_TYPE_LABELS,
  CONTROLNET_TYPES_BY_ARCH,
  PREPROCESSORS,
} from "../../shared/constants/controlnet";
import { formatPreprocessorName } from "../../shared/utils";
import { ImageThumbnail } from "./ImageThumbnail";

// Get preprocessors filtered by architecture
function getPreprocessorsForArch(
  arch: Architecture | null
): PreprocessorInfo[] {
  if (!arch) return [];
  const archKey = arch as "sd15" | "sdxl";
  return PREPROCESSORS.filter((p) => p.types[archKey] !== undefined);
}

// Get the ControlNet type for a preprocessor on a specific architecture
function getTypeForPreprocessor(
  preprocessorName: string,
  arch: Architecture | null
): ControlNetType | null {
  if (!arch) return null;
  const archKey = arch as "sd15" | "sdxl";
  const preprocessor = PREPROCESSORS.find((p) => p.name === preprocessorName);
  return preprocessor?.types[archKey] ?? null;
}

// Group preprocessors by category
function groupByCategory(
  preprocessors: PreprocessorInfo[]
): Map<string, PreprocessorInfo[]> {
  const grouped = new Map<string, PreprocessorInfo[]>();
  for (const p of preprocessors) {
    const list = grouped.get(p.category) ?? [];
    list.push(p);
    grouped.set(p.category, list);
  }
  return grouped;
}

// Encode type and preprocessor into a single value for the dropdown
function encodeValue(
  type: ControlNetType,
  preprocessor: string | null
): string {
  return preprocessor ? `${type}:${preprocessor}` : `${type}:`;
}

// Decode the dropdown value back to type and preprocessor
function decodeValue(value: string): {
  type: ControlNetType;
  preprocessor: string | null;
} {
  const [type, preprocessor] = value.split(":");
  return {
    type: type as ControlNetType,
    preprocessor: preprocessor || null,
  };
}

interface ControlNetSettingsProps {
  controlNets: ControlNetConfig[];
  architecture: Architecture | null;
  onAdd: (type: ControlNetType) => void;
  onUpdate: (index: number, updates: Partial<ControlNetConfig>) => void;
  onRemove: (index: number) => void;
  onPreprocess: (index: number) => void;
}

export function ControlNetSettings({
  controlNets,
  architecture,
  onAdd,
  onUpdate,
  onRemove,
  onPreprocess,
}: ControlNetSettingsProps) {
  // Get supported types for current architecture
  const supportedTypes = architecture
    ? (CONTROLNET_TYPES_BY_ARCH[architecture] ?? [])
    : [];

  // Only show for architectures that support ControlNet
  if (supportedTypes.length === 0) {
    return null;
  }

  // Get preprocessors filtered for current architecture
  const availablePreprocessors = getPreprocessorsForArch(architecture);
  const groupedPreprocessors = groupByCategory(availablePreprocessors);

  // Default type when adding new ControlNet
  const defaultType: ControlNetType =
    architecture === "sd15" ? "canny" : "lineart";

  return (
    <Box>
      <Flex align="center">
        <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
          ControlNet
        </Text>
        <Flex align="center" justify="between" style={{ flex: 3 }}>
          {controlNets.length === 0 && (
            <Text size="2" color="gray">
              None
            </Text>
          )}
          {controlNets.length > 0 && <Box />}
          <IconButton
            size="1"
            variant="ghost"
            onClick={() => onAdd(defaultType)}
          >
            <PlusIcon size={14} weight="bold" />
          </IconButton>
        </Flex>
      </Flex>

      {controlNets.map((config, index) => (
        <Flex
          key={index}
          gap="2"
          mt="2"
          p="2"
          className="rounded bg-[var(--gray-3)]"
        >
          {/* Image thumbnail on left */}
          <Box className="shrink-0">
            {config.image ? (
              <ImageThumbnail
                filename={config.image}
                context="controlnet"
                size="small"
                selectable={false}
                onRemove={() => onUpdate(index, { image: "" })}
              />
            ) : (
              <Box className="flex h-14 w-14 items-center justify-center rounded bg-[var(--gray-5)]">
                <Text size="1" color="gray">
                  No image
                </Text>
              </Box>
            )}
          </Box>

          {/* Controls on right */}
          <Flex direction="column" gap="1" style={{ flex: 1, minWidth: 0 }}>
            {/* Type/Preprocessor dropdown + Remove button */}
            <Flex align="center" gap="1">
              <Select.Root
                size="1"
                value={encodeValue(config.type, config.preprocessor)}
                onValueChange={(value) => {
                  const { type, preprocessor } = decodeValue(value);
                  // When selecting a preprocessor, also update the type based on architecture
                  const effectiveType = preprocessor
                    ? (getTypeForPreprocessor(preprocessor, architecture) ??
                      type)
                    : type;
                  onUpdate(index, { type: effectiveType, preprocessor });
                }}
              >
                <Select.Trigger style={{ flex: 1 }} />
                <Select.Content>
                  {/* Group preprocessors by category */}
                  {Array.from(groupedPreprocessors.entries()).map(
                    ([category, preprocessors]) => (
                      <Select.Group key={category}>
                        <Select.Label className="flex items-center gap-2">
                          <span>{category}</span>
                          <span className="h-px flex-1 bg-[var(--gray-6)]" />
                        </Select.Label>
                        {preprocessors.map((p) => {
                          const archKey = architecture as "sd15" | "sdxl";
                          const type = p.types[archKey]!;
                          return (
                            <Select.Item
                              key={p.name}
                              value={encodeValue(type, p.name)}
                            >
                              {formatPreprocessorName(p.name)}
                            </Select.Item>
                          );
                        })}
                      </Select.Group>
                    )
                  )}

                  {/* Preprocessed category - for already processed images */}
                  <Select.Group>
                    <Select.Label className="flex items-center gap-2">
                      <span>Preprocessed</span>
                      <span className="h-px flex-1 bg-[var(--gray-6)]" />
                    </Select.Label>
                    {supportedTypes.map((type) => (
                      <Select.Item key={type} value={encodeValue(type, null)}>
                        {CONTROLNET_TYPE_LABELS[type]} (Preprocessed)
                      </Select.Item>
                    ))}
                  </Select.Group>
                </Select.Content>
              </Select.Root>
              <IconButton
                size="1"
                variant="soft"
                color="gray"
                onClick={() => onRemove(index)}
              >
                <XIcon size={12} weight="bold" />
              </IconButton>
            </Flex>

            {/* Weight slider + Process button */}
            <Flex align="center" gap="1">
              <Slider
                size="1"
                value={[config.weight]}
                onValueChange={(value) => {
                  if (value[0] !== undefined) {
                    onUpdate(index, { weight: value[0] });
                  }
                }}
                min={0}
                max={2}
                step={0.05}
                style={{ flex: 1 }}
              />
              <Text size="1" color="gray" className="w-8 text-right">
                {config.weight.toFixed(2)}
              </Text>
              <Tooltip content="Run preprocessor">
                <IconButton
                  size="1"
                  variant="soft"
                  disabled={!config.preprocessor || !config.image}
                  onClick={() => onPreprocess(index)}
                >
                  <PlayIcon size={14} weight="fill" />
                </IconButton>
              </Tooltip>
            </Flex>
          </Flex>
        </Flex>
      ))}
    </Box>
  );
}
