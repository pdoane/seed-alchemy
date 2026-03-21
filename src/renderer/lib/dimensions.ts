// Dimension presets and helpers for architecture-aware sizing

import type { Architecture } from "../../shared/types/models";

export interface DimensionPreset {
  label: string;
  width: number;
  height: number;
}

// SD1.5 fixed presets (one edge at 512)
export const SD15_PRESETS: DimensionPreset[] = [
  { label: "1:4", width: 512, height: 2048 },
  { label: "1:3", width: 512, height: 1536 },
  { label: "1:2", width: 512, height: 1024 },
  { label: "2:3", width: 512, height: 768 },
  { label: "4:5", width: 512, height: 640 },
  { label: "1:1", width: 512, height: 512 },
  { label: "5:4", width: 640, height: 512 },
  { label: "3:2", width: 768, height: 512 },
  { label: "2:1", width: 1024, height: 512 },
  { label: "3:1", width: 1536, height: 512 },
  { label: "4:1", width: 2048, height: 512 },
];

// SDXL optimal presets (from training data)
// Portrait orientations (height > width)
export const SDXL_PRESETS: DimensionPreset[] = [
  { label: "1:4", width: 512, height: 2048 },
  { label: "1:3.9", width: 512, height: 1984 },
  { label: "1:3.75", width: 512, height: 1920 },
  { label: "1:3.6", width: 512, height: 1856 },
  { label: "1:3.1", width: 576, height: 1792 },
  { label: "1:3", width: 576, height: 1728 },
  { label: "1:2.9", width: 576, height: 1664 },
  { label: "2:5", width: 640, height: 1600 },
  { label: "1:2.4", width: 640, height: 1536 },
  { label: "1:2.1", width: 704, height: 1472 },
  { label: "1:2", width: 704, height: 1408 },
  { label: "1:1.9", width: 704, height: 1344 },
  { label: "4:7", width: 768, height: 1344 },
  { label: "3:5", width: 768, height: 1280 },
  { label: "2:3", width: 832, height: 1216 },
  { label: "5:7", width: 832, height: 1152 },
  { label: "7:9", width: 896, height: 1152 },
  { label: "4:5", width: 896, height: 1088 },
  { label: "8:9", width: 960, height: 1088 },
  { label: "15:16", width: 960, height: 1024 },
  // Square
  { label: "1:1", width: 1024, height: 1024 },
  // Landscape orientations (width > height)
  { label: "16:15", width: 1024, height: 960 },
  { label: "9:8", width: 1088, height: 960 },
  { label: "5:4", width: 1088, height: 896 },
  { label: "9:7", width: 1152, height: 896 },
  { label: "7:5", width: 1152, height: 832 },
  { label: "3:2", width: 1216, height: 832 },
  { label: "5:3", width: 1280, height: 768 },
  { label: "7:4", width: 1344, height: 768 },
  { label: "2:1", width: 1408, height: 704 },
  { label: "2.1:1", width: 1472, height: 704 },
  { label: "2.4:1", width: 1536, height: 640 },
  { label: "5:2", width: 1600, height: 640 },
  { label: "2.9:1", width: 1664, height: 576 },
  { label: "3:1", width: 1728, height: 576 },
  { label: "3.1:1", width: 1792, height: 576 },
  { label: "3.6:1", width: 1856, height: 512 },
  { label: "3.75:1", width: 1920, height: 512 },
  { label: "3.9:1", width: 1984, height: 512 },
  { label: "4:1", width: 2048, height: 512 },
];

// Aspect ratios for flexible mode (Flux, SD3, etc.)
export const ASPECT_RATIOS = [
  { label: "1:4", ratio: 0.25 },
  { label: "1:3", ratio: 1 / 3 },
  { label: "1:2", ratio: 0.5 },
  { label: "2:3", ratio: 2 / 3 },
  { label: "3:4", ratio: 0.75 },
  { label: "1:1", ratio: 1.0 },
  { label: "4:3", ratio: 4 / 3 },
  { label: "3:2", ratio: 1.5 },
  { label: "2:1", ratio: 2.0 },
  { label: "3:1", ratio: 3.0 },
  { label: "4:1", ratio: 4.0 },
] as const;

export const MEGAPIXELS = [0.5, 1.0, 1.5, 2.0] as const;
export type Megapixels = (typeof MEGAPIXELS)[number];

export const STEP_SIZE = 64;

// Deny list: architectures with fixed presets (do NOT support megapixels)
const FIXED_PRESET_ARCHITECTURES: Architecture[] = [
  "sd15",
  "sd20",
  "sd21",
  "sdxl",
  "sdxl-refiner",
];

export function supportsMegapixels(arch: Architecture | null): boolean {
  if (!arch) return true;
  return !FIXED_PRESET_ARCHITECTURES.includes(arch);
}

export function getPresets(
  arch: Architecture | null
): DimensionPreset[] | null {
  if (arch === "sd15" || arch === "sd20" || arch === "sd21")
    return SD15_PRESETS;
  if (arch === "sdxl" || arch === "sdxl-refiner") return SDXL_PRESETS;
  return null;
}

// Calculate dimensions from megapixels and ratio (for flexible architectures)
export function calculateDimensions(
  megapixels: number,
  ratio: number
): { width: number; height: number } {
  const totalPixels = megapixels * 1_000_000;
  // width = ratio * height, width * height = totalPixels
  // ratio * height^2 = totalPixels
  // height = sqrt(totalPixels / ratio)
  const height =
    Math.round(Math.sqrt(totalPixels / ratio) / STEP_SIZE) * STEP_SIZE;
  const width = Math.round((height * ratio) / STEP_SIZE) * STEP_SIZE;
  return { width, height };
}

// Find closest preset by aspect ratio
export function findClosestPreset(
  presets: DimensionPreset[],
  width: number,
  height: number
): DimensionPreset {
  const targetRatio = width / height;
  return presets.reduce((closest, preset) => {
    const presetRatio = preset.width / preset.height;
    const closestRatio = closest.width / closest.height;
    return Math.abs(presetRatio - targetRatio) <
      Math.abs(closestRatio - targetRatio)
      ? preset
      : closest;
  });
}

// Find preset index for a given width/height (exact match or closest)
export function findPresetIndex(
  presets: DimensionPreset[],
  width: number,
  height: number
): number {
  // Try exact match first
  const exactIndex = presets.findIndex(
    (p) => p.width === width && p.height === height
  );
  if (exactIndex !== -1) return exactIndex;

  // Fall back to closest by aspect ratio
  const closest = findClosestPreset(presets, width, height);
  return presets.indexOf(closest);
}

// Find closest aspect ratio for flexible mode
export function findClosestAspectRatio(
  width: number,
  height: number
): (typeof ASPECT_RATIOS)[number] {
  const targetRatio = width / height;
  return ASPECT_RATIOS.reduce((closest, ar) => {
    return Math.abs(ar.ratio - targetRatio) <
      Math.abs(closest.ratio - targetRatio)
      ? ar
      : closest;
  });
}

// Infer megapixels from dimensions
export function inferMegapixels(width: number, height: number): Megapixels {
  const totalPixels = width * height;
  return MEGAPIXELS.reduce((closest, mp) =>
    Math.abs(mp * 1_000_000 - totalPixels) <
    Math.abs(closest * 1_000_000 - totalPixels)
      ? mp
      : closest
  );
}

// Swap dimensions - for presets, find the swapped equivalent
export function swapDimensions(
  width: number,
  height: number,
  presets: DimensionPreset[] | null
): { width: number; height: number } {
  const swapped = { width: height, height: width };

  if (presets) {
    // Find closest preset to swapped dimensions
    const closest = findClosestPreset(presets, swapped.width, swapped.height);
    return { width: closest.width, height: closest.height };
  }

  return swapped;
}

// Apply aspect ratio from source dimensions to current dimensions
// For preset architectures: finds closest preset matching the source aspect ratio
// For flexible architectures: keeps current megapixels, applies source aspect ratio
export function applyAspectRatio(
  sourceWidth: number,
  sourceHeight: number,
  currentWidth: number,
  currentHeight: number,
  arch: Architecture | null
): { width: number; height: number } {
  const presets = getPresets(arch);

  if (presets) {
    const closest = findClosestPreset(presets, sourceWidth, sourceHeight);
    return { width: closest.width, height: closest.height };
  }

  if (supportsMegapixels(arch)) {
    const sourceRatio = findClosestAspectRatio(sourceWidth, sourceHeight);
    const currentMp = inferMegapixels(currentWidth, currentHeight);
    return calculateDimensions(currentMp, sourceRatio.ratio);
  }

  return { width: sourceWidth, height: sourceHeight };
}
