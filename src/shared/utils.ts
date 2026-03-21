// Shared utility functions

import type { Architecture, ModelInfo } from "./types/models.js";
import type { LoRAConfig } from "./types/image.js";
import type { ControlNetConfig } from "./types/controlnet.js";
import {
  CONTROLNET_TYPES_BY_ARCH,
  PREPROCESSORS,
} from "./constants/controlnet.js";

// Format preprocessor name for display (remove Preprocessor suffix and _aux)
export function formatPreprocessorName(name: string): string {
  return name
    .replace(/_Preprocessor$/, "")
    .replace(/Preprocessor$/, "")
    .replace(/_aux$/, "");
}

// Filter LoRAs by architecture compatibility
export function filterCompatibleLoras(
  loras: LoRAConfig[],
  arch: Architecture | null,
  availableLoras: ModelInfo[]
): { compatible: LoRAConfig[]; incompatible: string[] } {
  if (!arch || arch === "unknown") {
    return { compatible: loras, incompatible: [] };
  }

  const compatible: LoRAConfig[] = [];
  const incompatible: string[] = [];

  for (const lora of loras) {
    const loraInfo = availableLoras.find((l) => l.filename === lora.filename);
    if (
      !loraInfo ||
      loraInfo.architecture === "unknown" ||
      loraInfo.architecture === arch
    ) {
      compatible.push(lora);
    } else {
      incompatible.push(loraInfo.title || lora.filename);
    }
  }

  return { compatible, incompatible };
}

// Remap ControlNets for target architecture
export function remapControlNets(
  controlNets: ControlNetConfig[],
  arch: Architecture | null
): { remapped: ControlNetConfig[]; removedCount: number } {
  if (!arch || arch === "unknown") {
    return { remapped: controlNets, removedCount: 0 };
  }

  const supportedTypes = CONTROLNET_TYPES_BY_ARCH[arch] ?? [];
  if (supportedTypes.length === 0) {
    return { remapped: [], removedCount: controlNets.length };
  }

  const archKey = arch as "sd15" | "sdxl";
  const remapped: ControlNetConfig[] = [];

  for (const cn of controlNets) {
    if (cn.preprocessor) {
      const preprocessorInfo = PREPROCESSORS.find(
        (p) => p.name === cn.preprocessor
      );
      const newType = preprocessorInfo?.types[archKey];
      if (newType) {
        remapped.push({ ...cn, type: newType });
      }
    } else {
      if (supportedTypes.includes(cn.type)) {
        remapped.push(cn);
      }
    }
  }

  return { remapped, removedCount: controlNets.length - remapped.length };
}
