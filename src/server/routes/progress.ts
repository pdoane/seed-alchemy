import type { FastifyInstance } from "fastify";
import WebSocket from "ws";
import { randomUUID } from "crypto";
import { mkdir, writeFile, rename, access } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import type {
  ImageMetadata,
  OperationRecord,
} from "../../shared/types/image.js";
import type {
  ProgressEvent,
  ImageResult,
  GenerationSession,
  SessionResponse,
} from "../../shared/types/progress.js";
import { type HistoryResponse } from "../comfyui/index.js";
import { writePngMetadata } from "../png-metadata.js";
import { getUserImagesDir } from "../user-utils.js";
import { loadDocument, getDocumentPaths } from "../document-storage.js";

const COMFYUI_URL = process.env.COMFYUI_URL || "http://127.0.0.1:8000";
const COMFYUI_WS_URL = COMFYUI_URL.replace(/^http/, "ws");
const COMFYUI_OUTPUT_DIR =
  process.env.COMFYUI_OUTPUT_DIR || join(homedir(), "Documents/ComfyUI/output");

// Client ID for ComfyUI WebSocket - shared with prompt submissions
export const CLIENT_ID = randomUUID();

// Binary event types from ComfyUI
const BinaryEventTypes = {
  PREVIEW_IMAGE_WITH_METADATA: 4,
};

// Pending prompts waiting for completion (includes username for per-user image storage)
interface PendingPrompt {
  operation: OperationRecord;
  username: string;
  documentId?: string; // If set, save to document assets instead of global images
}
const pendingPrompts = new Map<string, PendingPrompt>();

// Current generation session state (for client reconnection)
let currentSession: GenerationSession | null = null;

// Start a new generation session (called from generate route)
export function startGenerationSession(
  promptId: string,
  width: number,
  height: number
): void {
  currentSession = {
    promptId,
    width,
    height,
    step: 0,
    maxSteps: 0,
    previewUrl: null,
    node: null,
  };
}

// Get current session for API endpoint
export function getCurrentSession(): SessionResponse {
  return { generation: currentSession };
}

let ws: WebSocket | null = null;
let reconnectTimeout: NodeJS.Timeout | null = null;

// SSE clients for progress updates
type SSEClient = {
  id: string;
  send: (data: string) => void;
};
const sseClients = new Set<SSEClient>();

function broadcastEvent(event: ProgressEvent) {
  const data = JSON.stringify(event);
  for (const client of sseClients) {
    try {
      client.send(`data: ${data}\n\n`);
    } catch {
      sseClients.delete(client);
    }
  }
}

function connectWebSocket() {
  if (ws?.readyState === WebSocket.OPEN) return;

  try {
    ws = new WebSocket(`${COMFYUI_WS_URL}/ws?clientId=${CLIENT_ID}`);

    ws.on("open", () => {
      console.log(`Connected to ComfyUI WebSocket (clientId: ${CLIENT_ID})`);

      // Send feature flags to enable preview metadata
      const featureFlags = {
        type: "feature_flags",
        data: {
          supports_preview_metadata: true,
        },
      };
      ws!.send(JSON.stringify(featureFlags));
    });

    ws.on("message", (data: WebSocket.Data) => {
      try {
        const event = parseMessage(data);
        if (event) {
          broadcastEvent(event);
        }
      } catch {
        // Ignore parse errors
      }
    });

    ws.on("close", () => {
      console.log("ComfyUI WebSocket closed, reconnecting...");
      scheduleReconnect();
    });

    ws.on("error", () => {
      console.log("ComfyUI WebSocket error");
      scheduleReconnect();
    });
  } catch {
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimeout) return;
  reconnectTimeout = setTimeout(() => {
    reconnectTimeout = null;
    connectWebSocket();
  }, 5000);
}

function parseMessage(data: WebSocket.Data): ProgressEvent | null {
  const buffer = Buffer.isBuffer(data)
    ? data
    : data instanceof ArrayBuffer
      ? Buffer.from(data)
      : null;

  if (buffer) {
    // Check if it's JSON sent as binary (starts with '{')
    if (buffer[0] === 0x7b) {
      const message = JSON.parse(buffer.toString());
      return handleJsonMessage(message);
    }

    // Handle binary preview images with metadata
    if (buffer.length >= 8) {
      const type = buffer.readUInt32BE(0);

      if (type === BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA) {
        const metadataLength = buffer.readUInt32BE(4);
        const metadataJson = buffer.subarray(8, 8 + metadataLength).toString();
        const metadata = JSON.parse(metadataJson) as { prompt_id: string };
        const imageData = buffer.subarray(8 + metadataLength);

        // Detect image type from magic bytes
        const isPng =
          imageData[0] === 0x89 &&
          imageData[1] === 0x50 &&
          imageData[2] === 0x4e &&
          imageData[3] === 0x47;
        const mimeType = isPng ? "image/png" : "image/jpeg";
        const previewUrl = `data:${mimeType};base64,${imageData.toString("base64")}`;

        // Update session state
        if (currentSession?.promptId === metadata.prompt_id) {
          currentSession.previewUrl = previewUrl;
        }

        return { type: "preview", promptId: metadata.prompt_id, previewUrl };
      }
    }
  } else {
    const message = JSON.parse(data.toString());
    return handleJsonMessage(message);
  }

  return null;
}

function handleJsonMessage(message: {
  type: string;
  data?: unknown;
}): ProgressEvent | null {
  switch (message.type) {
    case "execution_start": {
      const data = message.data as { prompt_id: string };
      return { type: "execution_start", promptId: data.prompt_id };
    }
    case "executing": {
      const data = message.data as { node: string | null; prompt_id: string };
      // The node ID is the key we used in the workflow (e.g., "sampler", "decode")
      const nodeName = data.node;
      // Update session state
      if (currentSession?.promptId === data.prompt_id) {
        currentSession.node = nodeName;
      }
      return { type: "executing", promptId: data.prompt_id, node: nodeName };
    }
    case "progress": {
      const data = message.data as {
        value: number;
        max: number;
        prompt_id: string;
      };
      // Update session state
      if (currentSession?.promptId === data.prompt_id) {
        currentSession.step = data.value;
        currentSession.maxSteps = data.max;
      }
      return {
        type: "progress",
        promptId: data.prompt_id,
        step: data.value,
        maxSteps: data.max,
      };
    }
    case "execution_success": {
      const data = message.data as { prompt_id: string };
      // Clear session state
      if (currentSession?.promptId === data.prompt_id) {
        currentSession = null;
      }
      // Process images asynchronously, don't block the WebSocket handler
      processCompletedImages(data.prompt_id);
      // Return null - we'll broadcast execution_success after processing images
      return null;
    }
    case "execution_interrupted": {
      const data = message.data as { prompt_id: string };
      // Clear session state
      if (currentSession?.promptId === data.prompt_id) {
        currentSession = null;
      }
      pendingPrompts.delete(data.prompt_id);
      return { type: "execution_interrupted", promptId: data.prompt_id };
    }
    case "execution_error": {
      const data = message.data as {
        prompt_id: string;
        exception_message?: string;
      };
      // Clear session state
      if (currentSession?.promptId === data.prompt_id) {
        currentSession = null;
      }
      pendingPrompts.delete(data.prompt_id);
      return {
        type: "execution_error",
        promptId: data.prompt_id,
        error: data.exception_message || "Unknown error",
      };
    }
    default:
      return null;
  }
}

// Image processing utilities
async function ensureImagesDir(username: string) {
  const imagesDir = getUserImagesDir(username);
  await mkdir(imagesDir, { recursive: true });
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function generateUniqueFilename(originalName: string): string {
  const ext = originalName.split(".").pop() || "png";
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `${timestamp}_${random}.${ext}`;
}

async function downloadImage(
  filename: string,
  subfolder: string,
  localPath: string
): Promise<void> {
  const comfyUrl = `${COMFYUI_URL}/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=output`;
  const imgResponse = await fetch(comfyUrl);

  if (imgResponse.ok) {
    const buffer = await imgResponse.arrayBuffer();
    await writeFile(localPath, Buffer.from(buffer));
  }
}

async function processCompletedImages(promptId: string): Promise<void> {
  try {
    const pending = pendingPrompts.get(promptId);
    if (!pending) {
      // No params stored - broadcast success without images
      broadcastEvent({ type: "execution_success", promptId, images: [] });
      return;
    }

    const { operation, username, documentId } = pending;

    const response = await fetch(`${COMFYUI_URL}/history/${promptId}`);
    if (!response.ok) {
      broadcastEvent({
        type: "execution_error",
        promptId,
        error: "Failed to fetch history",
      });
      pendingPrompts.delete(promptId);
      return;
    }

    const history = (await response.json()) as HistoryResponse;
    const promptHistory = history[promptId];

    if (!promptHistory) {
      broadcastEvent({
        type: "execution_error",
        promptId,
        error: "Prompt not found in history",
      });
      pendingPrompts.delete(promptId);
      return;
    }

    const images: ImageResult[] = [];
    const createdAt = new Date().toISOString();

    // Document mode: save to document assets folder
    if (documentId) {
      await processDocumentAssets(
        promptHistory,
        operation,
        username,
        documentId,
        createdAt,
        images
      );
    } else {
      // Image mode: save to global images folder
      await processGlobalImages(
        promptHistory,
        operation,
        username,
        createdAt,
        images
      );
    }

    pendingPrompts.delete(promptId);
    broadcastEvent({
      type: "execution_success",
      promptId,
      images,
      documentId,
    });
  } catch (error) {
    pendingPrompts.delete(promptId);
    broadcastEvent({
      type: "execution_error",
      promptId,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

// Process images for global images folder (Image Mode)
async function processGlobalImages(
  promptHistory: HistoryResponse[string],
  operation: OperationRecord,
  username: string,
  createdAt: string,
  images: ImageResult[]
): Promise<void> {
  const imagesDir = getUserImagesDir(username);
  await ensureImagesDir(username);

  for (const nodeOutput of Object.values(promptHistory.outputs)) {
    if (nodeOutput.images) {
      for (const img of nodeOutput.images) {
        const newFilename = generateUniqueFilename(img.filename);
        const localPath = join(imagesDir, newFilename);
        const comfyPath = join(COMFYUI_OUTPUT_DIR, img.subfolder, img.filename);

        // Try direct file move first
        if (await pathExists(comfyPath)) {
          try {
            await rename(comfyPath, localPath);
          } catch {
            // Fall through to API download
            await downloadImage(img.filename, img.subfolder, localPath);
          }
        } else {
          // Fallback: download via API
          await downloadImage(img.filename, img.subfolder, localPath);
        }

        // Embed metadata into PNG
        const metadata: ImageMetadata = {
          createdAt,
          operation,
        };
        await writePngMetadata(localPath, metadata);

        images.push({ filename: newFilename });
      }
    }
  }
}

// Process images for document assets folder (Document Mode)
async function processDocumentAssets(
  promptHistory: HistoryResponse[string],
  operation: OperationRecord,
  username: string,
  documentId: string,
  createdAt: string,
  images: ImageResult[]
): Promise<void> {
  const doc = await loadDocument(username, documentId);
  if (!doc) {
    throw new Error(`Document ${documentId} not found`);
  }

  const paths = getDocumentPaths(username, documentId);
  await mkdir(paths.assets, { recursive: true });

  for (const nodeOutput of Object.values(promptHistory.outputs)) {
    if (nodeOutput.images) {
      for (const img of nodeOutput.images) {
        const newFilename = generateUniqueFilename(img.filename);
        const localPath = join(paths.assets, newFilename);
        const comfyPath = join(COMFYUI_OUTPUT_DIR, img.subfolder, img.filename);

        // Try direct file move first
        if (await pathExists(comfyPath)) {
          try {
            await rename(comfyPath, localPath);
          } catch {
            await downloadImage(img.filename, img.subfolder, localPath);
          }
        } else {
          await downloadImage(img.filename, img.subfolder, localPath);
        }

        // Embed metadata into PNG
        const metadata: ImageMetadata = {
          createdAt,
          operation,
        };
        await writePngMetadata(localPath, metadata);

        images.push({ filename: newFilename, assetId: newFilename });
      }
    }
  }
}

// Called by generate/enhance routes to register pending prompt
export function registerPendingPrompt(
  promptId: string,
  operation: OperationRecord,
  username: string,
  documentId?: string
) {
  pendingPrompts.set(promptId, { operation, username, documentId });
}

export function registerProgressRoutes(fastify: FastifyInstance) {
  // Connect to ComfyUI WebSocket on startup
  connectWebSocket();

  // Get current generation session state (for client reconnection)
  fastify.get("/api/session", async () => {
    return getCurrentSession();
  });

  // SSE endpoint for real-time progress updates
  fastify.get("/api/progress/stream", async (request, reply) => {
    const clientId = randomUUID();

    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    });

    const client: SSEClient = {
      id: clientId,
      send: (data: string) => reply.raw.write(data),
    };

    sseClients.add(client);

    // Clean up on disconnect
    request.raw.on("close", () => {
      sseClients.delete(client);
    });

    // Keep connection open - don't return a response
    await new Promise(() => {});
  });

  // Cancel current generation
  fastify.delete("/api/generate", async (_request, reply) => {
    if (pendingPrompts.size === 0) {
      return reply.status(404).send({ error: "No generation in progress" });
    }

    try {
      // ComfyUI interrupt endpoint stops the current execution
      const response = await fetch(`${COMFYUI_URL}/interrupt`, {
        method: "POST",
      });

      if (!response.ok) {
        return reply.status(500).send({ error: "Failed to cancel generation" });
      }

      return { success: true };
    } catch (error) {
      fastify.log.error(error);
      return reply.status(500).send({ error: "Failed to cancel generation" });
    }
  });
}
