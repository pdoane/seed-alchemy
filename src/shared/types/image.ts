// Image types - generation parameters and provenance metadata

import type { ControlNetConfig } from "./controlnet.js";

// Blend modes for layer compositing
export type BlendMode =
  | "normal"
  | "multiply"
  | "screen"
  | "overlay"
  | "darken"
  | "lighten"
  | "color-dodge"
  | "color-burn"
  | "hard-light"
  | "soft-light"
  | "difference"
  | "exclusion";

// LoRA configuration (persisted)
export interface LoRAConfig {
  filename: string;
  weight: number;
}

// Reference image configuration
export interface ReferenceImageConfig {
  filename: string;
}

// Reference image weight type options
export type ReferenceWeightType =
  | "linear"
  | "ease in"
  | "ease out"
  | "ease in-out"
  | "reverse in-out"
  | "weak input"
  | "weak output"
  | "weak middle"
  | "strong middle"
  | "style transfer"
  | "composition"
  | "strong style transfer";

// Reference image combine mode options
export type ReferenceCombineMode =
  | "concat"
  | "add"
  | "subtract"
  | "average"
  | "norm average";

// Image generation parameters
export interface ImageParams {
  prompt: string;
  negativePrompt: string;
  model: string;
  sampler: string;
  scheduler: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  seed: number;
  loras: LoRAConfig[];
  sourceImage: string;
  sourceImageStrength: number;
  referenceImages: ReferenceImageConfig[];
  referenceWeight: number;
  referenceWeightType: ReferenceWeightType;
  referenceCombineMode: ReferenceCombineMode;
  controlNets: ControlNetConfig[];
  faceDetailer: boolean;
  upscaleEnabled: boolean;
  upscaleFactor: number;
}

// Snapshot of layer state at flatten time (for provenance)
export interface LayerSnapshot {
  assetId: string;
  opacity: number;
  blendMode: BlendMode;
  position: { x: number; y: number };
}

// Detector parameters for preprocessor operations
export interface DetectorParams {
  source: string;
  detector: string;
  resolution: number;
}

// Enhance operation type
export type EnhanceType = "face_detailer" | "upscale";

// Enhance operation parameters
export interface EnhanceParams {
  source: string;
  model: string;
  sampler: string;
  scheduler: string;
  steps: number;
  cfgScale: number;
  seed: number;
  loras: LoRAConfig[];
  prompt: string;
  negativePrompt: string;
  enhanceType: EnhanceType;
  upscaleFactor?: number;
}

// Non-sequence operation records
export type SingleOperationRecord =
  | { type: "raw" }
  | { type: "generate"; params: ImageParams }
  | { type: "detect"; params: DetectorParams }
  | { type: "flatten"; layers: LayerSnapshot[] }
  | { type: "enhance"; params: EnhanceParams };

// Operation records - what created an image
export type OperationRecord =
  | SingleOperationRecord
  | { type: "sequence"; operations: SingleOperationRecord[] };

// Image metadata embedded in PNG files
export interface ImageMetadata {
  createdAt: string;
  operation: OperationRecord;
}
