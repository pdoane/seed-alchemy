import type { ImageParams } from "./image";
import type { ToolType } from "./document";

export type AppMode = "image" | "document" | "canvas" | "gallery" | "models";

export interface ImageUiState {
  selectedFilename: string | null;
  seedLocked: boolean;
}

export interface DocumentUiState {
  currentDocumentId: string | null;
  activeTool: ToolType;
}

export interface UserState {
  mode: AppMode;
  params: ImageParams;
  ui: ImageUiState;
  documentUi: DocumentUiState;
}
