// Progress event types (SSE from server to client)

export interface ImageResult {
  filename: string;
  assetId?: string; // Present when generated for a document
}

export type ProgressEvent =
  | { type: "execution_start"; promptId: string }
  | { type: "executing"; promptId: string; node: string | null }
  | { type: "progress"; promptId: string; step: number; maxSteps: number }
  | { type: "preview"; promptId: string; previewUrl: string }
  | {
      type: "execution_success";
      promptId: string;
      images: ImageResult[];
      documentId?: string; // Present when generated for a document
    }
  | { type: "execution_interrupted"; promptId: string }
  | { type: "execution_error"; promptId: string; error: string };

// Generation session state (server-side, returned via GET /api/session)
export interface GenerationSession {
  promptId: string;
  width: number;
  height: number;
  step: number;
  maxSteps: number;
  previewUrl: string | null;
  node: string | null;
}

export interface SessionResponse {
  generation: GenerationSession | null;
}
