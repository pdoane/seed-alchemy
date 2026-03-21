// ControlNet types

// Superset of all types - each architecture supports a subset
export type ControlNetType =
  | "canny" // SD1.5 only
  | "openpose" // Both
  | "depth" // Both
  | "normal" // Both (normalbae in SD1.5)
  | "lineart" // Both
  | "lineart_anime" // SD1.5 only
  | "scribble" // Both
  | "softedge" // SD1.5 only (hed/pidinet)
  | "segment" // Both
  | "tile" // Both
  | "mlsd" // SD1.5 only
  | "shuffle" // SD1.5 only
  | "inpaint" // SD1.5 only
  | "repaint"; // SDXL only

// Preprocessor info with architecture-specific type mappings
export interface PreprocessorInfo {
  name: string;
  category: string;
  types: {
    sd15?: ControlNetType;
    sdxl?: ControlNetType;
  };
}

// ControlNet configuration
export interface ControlNetConfig {
  type: ControlNetType;
  image: string;
  preprocessor: string | null; // null = already preprocessed
  weight: number;
}
