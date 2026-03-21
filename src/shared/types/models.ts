// Model management types
// Keep ModelFolder in sync with MODEL_FOLDERS in constants/models.ts

export type ModelFolder =
  | "checkpoints"
  | "diffusion_models"
  | "loras"
  | "vae"
  | "controlnet"
  | "unet"
  | "clip"
  | "text_encoders"
  | "clip_vision"
  | "embeddings"
  | "upscale_models"
  | "ipadapter";

export type ArchitectureSource = "modelspec" | "kohya" | "tensor" | "unknown";

export type Architecture =
  | "sd15"
  | "sd20"
  | "sd21"
  | "sdxl"
  | "sdxl-refiner"
  | "sd3"
  | "sd35"
  | "flux"
  | "cascade"
  | "wan"
  | "zit"
  | "unknown";

export interface ModelInfo {
  filename: string;
  path: string;
  folder: ModelFolder;
  tensorCount: number;
  fileSizeBytes: number;
  architecture: Architecture;
  architectureSource: ArchitectureSource;
  title: string; // Display name - from metadata or derived from filename
  author?: string;
  description?: string;
  thumbnail?: string;
  license?: string;
  hash?: string;
}
