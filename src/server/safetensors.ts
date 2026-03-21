import { open, stat } from "node:fs/promises";
import type {
  Architecture,
  ArchitectureSource,
} from "../shared/types/models.js";

export interface SafetensorsHeader {
  tensors: Record<string, TensorInfo>;
  metadata: Record<string, string>;
}

export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface ArchitectureResult {
  architecture: Architecture;
  source: ArchitectureSource;
}

// Read and parse the header from a safetensors file.
// The format is:
// - 8 bytes: little-endian uint64 header length
// - N bytes: JSON header containing tensor info and __metadata__
export async function readSafetensorsHeader(
  filePath: string
): Promise<SafetensorsHeader> {
  const fd = await open(filePath, "r");
  try {
    // Read the 8-byte header length
    const lengthBuffer = Buffer.alloc(8);
    await fd.read(lengthBuffer, 0, 8, 0);
    const headerLength = lengthBuffer.readBigUInt64LE();

    // Sanity check - headers shouldn't be larger than 100MB
    if (headerLength > 100_000_000n) {
      throw new Error(`Header length too large: ${headerLength}`);
    }

    // Read the JSON header
    const headerBuffer = Buffer.alloc(Number(headerLength));
    await fd.read(headerBuffer, 0, Number(headerLength), 8);
    const headerJson = headerBuffer.toString("utf-8");
    const header = JSON.parse(headerJson);

    // Extract __metadata__ if present
    const metadata: Record<string, string> = header.__metadata__ || {};
    delete header.__metadata__;

    // Remaining keys are tensor definitions
    const tensors: Record<string, TensorInfo> = header;

    return { tensors, metadata };
  } finally {
    await fd.close();
  }
}

// Get file size in bytes
export async function getFileSize(filePath: string): Promise<number> {
  const stats = await stat(filePath);
  return stats.size;
}

// Detect model architecture from safetensors header.
// Priority order:
// 1. modelspec.architecture metadata
// 2. Kohya ss_base_model_version metadata
// 3. Tensor shape inspection
export function detectArchitecture(
  header: SafetensorsHeader
): ArchitectureResult {
  // Try modelspec.architecture first
  const modelspec = header.metadata["modelspec.architecture"];
  if (modelspec) {
    const arch = parseModelspecArchitecture(modelspec);
    if (arch !== "unknown") {
      return { architecture: arch, source: "modelspec" };
    }
  }

  // Try Kohya ss_base_model_version
  const kohya = header.metadata["ss_base_model_version"];
  if (kohya) {
    const arch = parseKohyaArchitecture(kohya);
    if (arch !== "unknown") {
      return { architecture: arch, source: "kohya" };
    }
  }

  // Fallback to tensor inspection
  const arch = detectArchitectureFromTensors(header.tensors);
  if (arch !== "unknown") {
    return { architecture: arch, source: "tensor" };
  }

  return { architecture: "unknown", source: "unknown" };
}

// Parse Stability AI modelspec.architecture format
function parseModelspecArchitecture(value: string): Architecture {
  const lower = value.toLowerCase();

  if (lower.includes("stable-diffusion-xl") && lower.includes("refiner")) {
    return "sdxl-refiner";
  }
  if (lower.includes("stable-diffusion-xl") || lower.includes("sdxl")) {
    return "sdxl";
  }
  if (lower.includes("sd-3.5") || lower.includes("sd3.5")) {
    return "sd35";
  }
  if (lower.includes("sd-3") || lower.includes("sd3")) {
    return "sd3";
  }
  if (lower.includes("flux")) {
    return "flux";
  }
  if (lower.includes("stable-cascade") || lower.includes("cascade")) {
    return "cascade";
  }
  if (lower.includes("sd-2.1") || lower.includes("sd2.1")) {
    return "sd21";
  }
  if (lower.includes("sd-2") || lower.includes("sd2")) {
    return "sd20";
  }
  if (
    lower.includes("stable-diffusion-v1") ||
    lower.includes("sd-1") ||
    lower.includes("sd1")
  ) {
    return "sd15";
  }

  return "unknown";
}

// Parse Kohya ss_base_model_version format
function parseKohyaArchitecture(value: string): Architecture {
  const lower = value.toLowerCase();

  if (lower.includes("xl")) {
    return "sdxl";
  }
  if (lower.includes("sd_v2") || lower.includes("sd2")) {
    if (lower.includes("768")) {
      return "sd21";
    }
    return "sd20";
  }
  if (
    lower.includes("sd_v1") ||
    lower.includes("sd1") ||
    lower.includes("v1-5") ||
    lower.includes("1.5")
  ) {
    return "sd15";
  }

  return "unknown";
}

// Detect architecture by inspecting tensor names and shapes
function detectArchitectureFromTensors(
  tensors: Record<string, TensorInfo>
): Architecture {
  const tensorNames = Object.keys(tensors);

  // Check for ZIT (Z-Image Turbo) tensors
  if (
    tensorNames.some((name) => name.startsWith("context_refiner.0.attention"))
  ) {
    return "zit";
  }

  // Check for Flux-specific tensors
  if (
    tensorNames.some((name) =>
      name.includes("double_blocks.0.img_attn.norm.key_norm.scale")
    )
  ) {
    return "flux";
  }

  // Check for SD3-specific tensors
  if (
    tensorNames.some((name) =>
      name.includes("joint_blocks.0.context_block.attn.qkv.weight")
    )
  ) {
    // Could be SD3 or SD3.5 - hard to distinguish without metadata
    return "sd3";
  }

  // Check for Stable Cascade
  if (tensorNames.some((name) => name.includes("clf.1.weight"))) {
    return "cascade";
  }

  // Check for Wan/Video models
  if (tensorNames.some((name) => name.match(/diffusion_model\.blocks\.\d+/))) {
    return "wan";
  }

  // Check for LoRA-specific patterns
  if (tensorNames.some((name) => name.startsWith("lora_"))) {
    return detectLoraArchitecture(tensorNames);
  }

  // Check SD1.x/SD2.x/SDXL by context dimension in cross-attention
  const contextDim = findContextDimension(tensors);
  if (contextDim !== null) {
    if (contextDim === 768) {
      return "sd15";
    }
    if (contextDim === 1024) {
      return "sd20";
    }
    if (contextDim === 2048) {
      return "sdxl";
    }
    if (contextDim === 1280) {
      return "sdxl-refiner";
    }
  }

  return "unknown";
}

// Detect LoRA architecture from tensor names
function detectLoraArchitecture(tensorNames: string[]): Architecture {
  const hasTE1 = tensorNames.some((name) => name.includes("lora_te1_"));
  const hasTE2 = tensorNames.some((name) => name.includes("lora_te2_"));
  const hasTE = tensorNames.some(
    (name) => name.includes("lora_te_") && !name.includes("lora_te1_")
  );

  // SDXL uses dual text encoders (te1 and te2)
  if (hasTE1 && hasTE2) {
    return "sdxl";
  }

  // SD1.x/SD2.x uses single text encoder
  if (hasTE && !hasTE1 && !hasTE2) {
    return "sd15"; // Could be SD2, but SD1.5 is more common
  }

  return "unknown";
}

// Find the context dimension from cross-attention tensors
function findContextDimension(
  tensors: Record<string, TensorInfo>
): number | null {
  // Look for attn2.to_k.weight which is the cross-attention key projection
  for (const [name, info] of Object.entries(tensors)) {
    if (name.includes("attn2") && name.includes("to_k") && info.shape) {
      // Shape is typically [hidden_dim, context_dim]
      if (info.shape.length >= 2) {
        const contextDim = info.shape[info.shape.length - 1];
        if (contextDim !== undefined) {
          return contextDim;
        }
      }
    }
  }
  return null;
}
