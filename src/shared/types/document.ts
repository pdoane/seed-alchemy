// Document mode types - Asset-centric model
// Layers are typeless and reference assets. Assets store provenance.

import type { BlendMode, ImageMetadata } from "./image.js";

// Tool types for editing
export type ToolType =
  | "select"
  | "rectangle"
  | "lasso"
  | "wand"
  | "brush"
  | "eraser"
  | "face"
  | "scribble";

// Asset definition - immutable images with provenance
export interface Asset {
  filename: string;
  metadata: ImageMetadata;
}

// Layer definition - typeless, references an asset
export interface Layer {
  id: string;
  name: string;
  asset?: string; // Reference to asset id (optional - can be empty layer)
  visible: boolean;
  opacity: number; // 0-100
  blendMode: BlendMode;
  position: { x: number; y: number };
  mask?: string; // Optional mask filename in layers/ folder
}

// Layer group for hierarchical organization
export interface LayerGroup {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  blendMode: BlendMode;
  children: (Layer | LayerGroup)[];
}

// Type guard for layer vs layer group
export function isLayerGroup(item: Layer | LayerGroup): item is LayerGroup {
  return "children" in item;
}

// Canvas size
export interface CanvasSize {
  width: number;
  height: number;
}

// Full document definition
export interface Document {
  id: string;
  name?: string;
  createdAt: string;
  modifiedAt: string;
  canvasSize: CanvasSize;
  layers: (Layer | LayerGroup)[];
  assets: Asset[];
  selectedLayerId: string | null;
}

// Document summary for listing (without full layer details)
export interface DocumentSummary {
  id: string;
  name?: string;
  modifiedAt: string;
  canvasSize: CanvasSize;
  layerCount: number;
  thumbnail?: string; // Path to composite.png if it exists
}

// API request/response types
export interface CreateDocumentRequest {
  name?: string;
  canvasSize: CanvasSize;
}

export interface UpdateDocumentRequest {
  name?: string;
  canvasSize?: CanvasSize;
  layers?: (Layer | LayerGroup)[];
  selectedLayerId?: string | null;
}
