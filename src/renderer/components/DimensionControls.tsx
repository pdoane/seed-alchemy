import { useState, useCallback } from "react";
import {
  Box,
  Button,
  Flex,
  IconButton,
  Slider,
  Text,
  Tooltip,
} from "@radix-ui/themes";
import { ArrowsLeftRightIcon, GearIcon } from "@phosphor-icons/react";
import type { Architecture } from "../../shared/types/models";
import {
  ASPECT_RATIOS,
  MEGAPIXELS,
  STEP_SIZE,
  type Megapixels,
  getPresets,
  supportsMegapixels,
  findPresetIndex,
  findClosestAspectRatio,
  inferMegapixels,
  calculateDimensions,
  swapDimensions,
} from "../lib/dimensions";

const MIN_DIMENSION = 64;
const MAX_DIMENSION = 4096;

interface DimensionControlsProps {
  width: number;
  height: number;
  architecture: Architecture | null;
  onChange: (width: number, height: number) => void;
}

export function DimensionControls({
  width,
  height,
  architecture,
  onChange,
}: DimensionControlsProps) {
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [customWidth, setCustomWidth] = useState(width);
  const [customHeight, setCustomHeight] = useState(height);

  const presets = getPresets(architecture);
  const showMegapixels = supportsMegapixels(architecture);

  // For preset mode
  const presetIndex = presets ? findPresetIndex(presets, width, height) : 0;
  const currentPresetLabel = presets?.[presetIndex]?.label ?? "";

  // For flexible mode
  const currentRatio = findClosestAspectRatio(width, height);
  const currentMegapixels = inferMegapixels(width, height);
  const ratioIndex = ASPECT_RATIOS.findIndex(
    (ar) => ar.label === currentRatio.label
  );

  const handlePresetChange = useCallback(
    (index: number) => {
      if (presets && presets[index]) {
        onChange(presets[index].width, presets[index].height);
      }
    },
    [presets, onChange]
  );

  const handleRatioChange = useCallback(
    (index: number) => {
      const ratio = ASPECT_RATIOS[index];
      if (ratio) {
        const dims = calculateDimensions(currentMegapixels, ratio.ratio);
        onChange(dims.width, dims.height);
      }
    },
    [currentMegapixels, onChange]
  );

  const handleMegapixelChange = useCallback(
    (mp: Megapixels) => {
      const dims = calculateDimensions(mp, currentRatio.ratio);
      onChange(dims.width, dims.height);
    },
    [currentRatio, onChange]
  );

  const handleSwap = useCallback(() => {
    const swapped = swapDimensions(width, height, presets);
    onChange(swapped.width, swapped.height);
  }, [width, height, presets, onChange]);

  const handleCustomApply = useCallback(() => {
    onChange(customWidth, customHeight);
    setIsCustomMode(false);
  }, [customWidth, customHeight, onChange]);

  const handleCustomCancel = useCallback(() => {
    setCustomWidth(width);
    setCustomHeight(height);
    setIsCustomMode(false);
  }, [width, height]);

  const enterCustomMode = useCallback(() => {
    setCustomWidth(width);
    setCustomHeight(height);
    setIsCustomMode(true);
  }, [width, height]);

  // Display values - show custom values when in custom mode
  const displayWidth = isCustomMode ? customWidth : width;
  const displayHeight = isCustomMode ? customHeight : height;
  const displayRatio = findClosestAspectRatio(displayWidth, displayHeight);
  const aspectLabel = presets ? currentPresetLabel : displayRatio.label;

  return (
    <Box>
      {/* Header line: Label, dimensions, aspect ratio, swap, custom */}
      <Flex align="center" mb="2">
        <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
          Size
        </Text>
        <Flex align="center" gap="2" style={{ flex: 3 }}>
          <Text size="2" color="gray">
            {displayWidth}×{displayHeight}
          </Text>
          <Text size="2" color="gray">
            {isCustomMode ? displayRatio.label : aspectLabel}
          </Text>
          <Box style={{ flex: 1 }} />
          <Tooltip content="Swap width and height">
            <IconButton size="1" variant="ghost" onClick={handleSwap}>
              <ArrowsLeftRightIcon size={14} weight="bold" />
            </IconButton>
          </Tooltip>
          <Tooltip content="Custom dimensions">
            <IconButton
              size="1"
              variant={isCustomMode ? "solid" : "ghost"}
              onClick={() =>
                isCustomMode ? handleCustomCancel() : enterCustomMode()
              }
            >
              <GearIcon size={14} weight="fill" />
            </IconButton>
          </Tooltip>
        </Flex>
      </Flex>

      {isCustomMode ? (
        // Custom input mode with sliders
        <Flex direction="column" gap="2">
          <Box>
            <Flex justify="between" mb="1">
              <Text size="1" color="gray">
                Width
              </Text>
              <Text size="1" color="gray">
                {customWidth}
              </Text>
            </Flex>
            <Slider
              size="1"
              value={[customWidth]}
              onValueChange={(value) => setCustomWidth(value[0]!)}
              min={MIN_DIMENSION}
              max={MAX_DIMENSION}
              step={STEP_SIZE}
            />
          </Box>
          <Box>
            <Flex justify="between" mb="1">
              <Text size="1" color="gray">
                Height
              </Text>
              <Text size="1" color="gray">
                {customHeight}
              </Text>
            </Flex>
            <Slider
              size="1"
              value={[customHeight]}
              onValueChange={(value) => setCustomHeight(value[0]!)}
              min={MIN_DIMENSION}
              max={MAX_DIMENSION}
              step={STEP_SIZE}
            />
          </Box>
          <Flex gap="2" justify="end">
            <Button size="1" variant="soft" onClick={handleCustomApply}>
              Apply
            </Button>
          </Flex>
        </Flex>
      ) : presets ? (
        // Fixed preset mode (SD1.5, SDXL)
        <Box>
          <Slider
            size="1"
            value={[presetIndex]}
            onValueChange={(value) => handlePresetChange(value[0]!)}
            min={0}
            max={presets.length - 1}
            step={1}
          />
          <Flex justify="between" mt="1">
            <Text size="1" color="gray">
              {presets[0]?.label}
            </Text>
            <Text size="1" color="gray">
              {presets[Math.floor(presets.length / 2)]?.label}
            </Text>
            <Text size="1" color="gray">
              {presets[presets.length - 1]?.label}
            </Text>
          </Flex>
        </Box>
      ) : (
        // Flexible mode (Flux, SD3, etc.)
        <Flex direction="column" gap="2">
          {showMegapixels && (
            <Flex gap="1">
              {MEGAPIXELS.map((mp) => (
                <Button
                  key={mp}
                  size="1"
                  variant={currentMegapixels === mp ? "solid" : "soft"}
                  color={currentMegapixels === mp ? "blue" : "gray"}
                  onClick={() => handleMegapixelChange(mp)}
                  style={{ flex: 1 }}
                >
                  {mp} MP
                </Button>
              ))}
            </Flex>
          )}
          <Box>
            <Slider
              size="1"
              value={[ratioIndex]}
              onValueChange={(value) => handleRatioChange(value[0]!)}
              min={0}
              max={ASPECT_RATIOS.length - 1}
              step={1}
            />
            <Flex justify="between" mt="1">
              <Text size="1" color="gray">
                {ASPECT_RATIOS[0]?.label}
              </Text>
              <Text size="1" color="gray">
                {ASPECT_RATIOS[Math.floor(ASPECT_RATIOS.length / 2)]?.label}
              </Text>
              <Text size="1" color="gray">
                {ASPECT_RATIOS[ASPECT_RATIOS.length - 1]?.label}
              </Text>
            </Flex>
          </Box>
        </Flex>
      )}
    </Box>
  );
}
