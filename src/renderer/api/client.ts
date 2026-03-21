import type {
  Document,
  DocumentSummary,
  CanvasSize,
  Layer,
  LayerGroup,
} from "../../shared/types/document";
import type {
  EnhanceParams,
  ImageMetadata,
  ImageParams,
} from "../../shared/types/image";
import type { ModelInfo, ModelFolder } from "../../shared/types/models";
import type {
  ProgressEvent,
  SessionResponse,
} from "../../shared/types/progress";
import type { UserState } from "../../shared/types/state";
import type { User } from "../../shared/types/users";

export interface SessionInfo {
  authenticated: boolean;
  username?: string;
}

const API_BASE = import.meta.env.DEV ? "http://localhost:3030" : "";

interface GenerateResponse {
  promptId: string;
}

interface PreprocessResponse {
  promptId: string;
}

interface EnhanceResponse {
  promptId: string;
}

interface FolderCount {
  folder: ModelFolder;
  count: number;
}

interface ImagesResponse {
  filenames: string[];
}

export const api = {
  async get<T>(path: string): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch ${path}`);
    }

    return response.json();
  },

  // Auth session management
  async login(username: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/auth`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Login failed");
    }
  },

  async logout(): Promise<void> {
    const response = await fetch(`${API_BASE}/api/auth`, {
      method: "DELETE",
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Logout failed");
    }
  },

  async getAuthSession(): Promise<SessionInfo> {
    const response = await fetch(`${API_BASE}/api/auth`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch auth session");
    }

    return response.json();
  },

  // Generation session (for reconnecting to in-progress generation)
  async getGenerationSession(): Promise<SessionResponse> {
    const response = await fetch(`${API_BASE}/api/session`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch generation session");
    }

    return response.json();
  },

  async generate(
    params: ImageParams,
    imageCount: number = 1,
    documentId?: string
  ): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, imageCount, documentId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Generation failed");
    }

    return response.json();
  },

  async enhance(
    params: EnhanceParams,
    documentId?: string
  ): Promise<EnhanceResponse> {
    const response = await fetch(`${API_BASE}/api/enhance`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, documentId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Enhance failed");
    }

    return response.json();
  },

  // Get all models, optionally filtered by folder
  async getModels(folder?: ModelFolder): Promise<ModelInfo[]> {
    const url = folder
      ? `${API_BASE}/api/models?folder=${folder}`
      : `${API_BASE}/api/models`;
    const response = await fetch(url, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch models");
    }

    return response.json();
  },

  // Get checkpoint names for the model dropdown (convenience method)
  async getCheckpointNames(): Promise<string[]> {
    const models = await this.getModels("checkpoints");
    return models.map((m) => m.filename);
  },

  // Get LoRA names for the LoRA selector (convenience method)
  async getLoraNames(): Promise<string[]> {
    const models = await this.getModels("loras");
    return models.map((m) => m.filename);
  },

  // Get available model folders with counts
  async getModelFolders(): Promise<FolderCount[]> {
    const response = await fetch(`${API_BASE}/api/models/folders`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch model folders");
    }

    return response.json();
  },

  getImageUrl(filename: string): string {
    return `${API_BASE}/api/images/${encodeURIComponent(filename)}`;
  },

  async getImages(): Promise<string[]> {
    const response = await fetch(`${API_BASE}/api/images`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch images");
    }

    const data: ImagesResponse = await response.json();
    return data.filenames;
  },

  async getImageMetadata(filename: string): Promise<ImageMetadata> {
    const response = await fetch(
      `${API_BASE}/api/images/${encodeURIComponent(filename)}/metadata`,
      {
        credentials: "include",
      }
    );

    if (!response.ok) {
      throw new Error("Failed to fetch image metadata");
    }

    return response.json();
  },

  async deleteImage(filename: string): Promise<void> {
    const response = await fetch(
      `${API_BASE}/api/images/${encodeURIComponent(filename)}`,
      {
        method: "DELETE",
        credentials: "include",
      }
    );

    if (!response.ok) {
      throw new Error("Failed to delete image");
    }
  },

  async getState(): Promise<UserState | null> {
    const response = await fetch(`${API_BASE}/api/state`, {
      credentials: "include",
    });

    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      throw new Error("Failed to fetch state");
    }

    return response.json();
  },

  async saveState(state: UserState): Promise<void> {
    const response = await fetch(`${API_BASE}/api/state`, {
      method: "PUT",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
    });

    if (!response.ok) {
      throw new Error("Failed to save state");
    }
  },

  // Subscribe to progress events via SSE
  subscribeProgress(onEvent: (event: ProgressEvent) => void): () => void {
    const eventSource = new EventSource(`${API_BASE}/api/progress/stream`);

    eventSource.onmessage = (event) => {
      try {
        const progressEvent = JSON.parse(event.data) as ProgressEvent;
        onEvent(progressEvent);
      } catch {
        // Ignore parse errors
      }
    };

    eventSource.onerror = () => {
      // EventSource automatically reconnects on error
    };

    // Return cleanup function
    return () => eventSource.close();
  },

  async cancelGeneration(): Promise<void> {
    const response = await fetch(`${API_BASE}/api/generate`, {
      method: "DELETE",
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to cancel generation");
    }
  },

  // Delete a model file
  async deleteModel(folder: ModelFolder, filename: string): Promise<void> {
    const response = await fetch(
      `${API_BASE}/api/models/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`,
      {
        method: "DELETE",
        credentials: "include",
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to delete model");
    }
  },

  // Upload a model file
  async uploadModel(
    file: File,
    targetFolder: ModelFolder,
    onProgress?: (progress: number) => void
  ): Promise<ModelInfo> {
    const formData = new FormData();
    // IMPORTANT: targetFolder must be appended BEFORE file
    // because @fastify/multipart only sees fields that come before the file
    formData.append("targetFolder", targetFolder);
    formData.append("file", file);

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = (event.loaded / event.total) * 100;
          onProgress(progress);
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.error || "Failed to upload model"));
          } catch {
            reject(new Error("Failed to upload model"));
          }
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Network error during upload"));
      });

      xhr.addEventListener("abort", () => {
        reject(new Error("Upload cancelled"));
      });

      xhr.open("POST", `${API_BASE}/api/models/upload`);
      xhr.withCredentials = true;
      xhr.send(formData);
    });
  },

  // User management
  async getUsers(): Promise<User[]> {
    const response = await fetch(`${API_BASE}/api/users`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch users");
    }

    return response.json();
  },

  async createUser(username: string, password?: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/users`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to create user");
    }
  },

  async deleteUser(username: string): Promise<void> {
    const response = await fetch(
      `${API_BASE}/api/users/${encodeURIComponent(username)}`,
      {
        method: "DELETE",
        credentials: "include",
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to delete user");
    }
  },

  async verifyPassword(username: string, password: string): Promise<boolean> {
    const response = await fetch(
      `${API_BASE}/api/users/${encodeURIComponent(username)}/verify`,
      {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      }
    );

    if (!response.ok) {
      throw new Error("Failed to verify password");
    }

    const result = await response.json();
    return result.valid;
  },

  // Preprocess an image with a ControlNet preprocessor
  async preprocess(
    image: string,
    preprocessor: string,
    resolution: number
  ): Promise<PreprocessResponse> {
    const response = await fetch(`${API_BASE}/api/preprocess`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, preprocessor, resolution }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Preprocessing failed");
    }

    return response.json();
  },

  // Document API
  async getDocuments(): Promise<DocumentSummary[]> {
    const response = await fetch(`${API_BASE}/api/documents`, {
      credentials: "include",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch documents");
    }

    const data = await response.json();
    return data.documents;
  },

  async getDocument(id: string): Promise<Document> {
    const response = await fetch(
      `${API_BASE}/api/documents/${encodeURIComponent(id)}`,
      {
        credentials: "include",
      }
    );

    if (!response.ok) {
      throw new Error("Failed to fetch document");
    }

    return response.json();
  },

  async createDocument(
    name?: string,
    canvasSize?: CanvasSize
  ): Promise<Document> {
    const response = await fetch(`${API_BASE}/api/documents`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, canvasSize }),
    });

    if (!response.ok) {
      throw new Error("Failed to create document");
    }

    return response.json();
  },

  async updateDocument(
    id: string,
    updates: {
      name?: string;
      canvasSize?: CanvasSize;
      layers?: (Layer | LayerGroup)[];
      selectedLayerId?: string | null;
    }
  ): Promise<Document> {
    const response = await fetch(
      `${API_BASE}/api/documents/${encodeURIComponent(id)}`,
      {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      }
    );

    if (!response.ok) {
      throw new Error("Failed to update document");
    }

    return response.json();
  },

  async deleteDocument(id: string): Promise<void> {
    const response = await fetch(
      `${API_BASE}/api/documents/${encodeURIComponent(id)}`,
      {
        method: "DELETE",
        credentials: "include",
      }
    );

    if (!response.ok) {
      throw new Error("Failed to delete document");
    }
  },

  getDocumentCompositeUrl(id: string): string {
    return `${API_BASE}/api/documents/${encodeURIComponent(id)}/composite`;
  },

  getDocumentLayerUrl(id: string, filename: string): string {
    return `${API_BASE}/api/documents/${encodeURIComponent(id)}/layers/${encodeURIComponent(filename)}`;
  },

  getDocumentAssetUrl(id: string, filename: string): string {
    return `${API_BASE}/api/documents/${encodeURIComponent(id)}/assets/${encodeURIComponent(filename)}`;
  },
};
