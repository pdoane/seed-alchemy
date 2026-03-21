// Document storage utilities
// Handles reading/writing document.json files and managing document directories

import { mkdir, readdir, readFile, rm, stat, writeFile } from "fs/promises";
import { join } from "path";
import { randomUUID } from "crypto";
import type {
  Document,
  DocumentSummary,
  CanvasSize,
  Layer,
  LayerGroup,
  Asset,
} from "../shared/types/document.js";
import { getDocumentDir, getUserDocumentsDir } from "./user-utils.js";
import { readPngMetadata } from "./png-metadata.js";

// Document directory structure:
// documents/<uuid>/
//   document.json    - Document metadata and layer stack
//   layers/          - Layer output images
//   assets/          - Auxiliary outputs (preprocessor results, etc.)
//   composite.png    - Current combined result

const DOCUMENT_JSON = "document.json";
const LAYERS_DIR = "layers";
const ASSETS_DIR = "assets";
const COMPOSITE_FILE = "composite.png";

// On-disk format (assets are stored in PNG files, not here)
interface DocumentOnDisk {
  id: string;
  name?: string;
  createdAt: string;
  modifiedAt: string;
  canvasSize: CanvasSize;
  layers: (Layer | LayerGroup)[];
  selectedLayerId: string | null;
}

// Create a new document with default values
export function createDocument(
  name?: string,
  canvasSize?: CanvasSize
): Document {
  const now = new Date().toISOString();
  return {
    id: randomUUID(),
    name,
    createdAt: now,
    modifiedAt: now,
    canvasSize: canvasSize ?? { width: 1024, height: 1024 },
    layers: [],
    assets: [],
    selectedLayerId: null,
  };
}

// Initialize document directory structure
async function initDocumentDir(docDir: string): Promise<void> {
  await mkdir(docDir, { recursive: true });
  await mkdir(join(docDir, LAYERS_DIR), { recursive: true });
  await mkdir(join(docDir, ASSETS_DIR), { recursive: true });
}

// Save document to disk (assets stored in PNG files, not here)
export async function saveDocument(
  username: string,
  document: Document
): Promise<void> {
  const docDir = getDocumentDir(username, document.id);
  await initDocumentDir(docDir);

  // Update modification time
  document.modifiedAt = new Date().toISOString();

  // Only save the on-disk format (no assets array)
  const onDisk: DocumentOnDisk = {
    id: document.id,
    name: document.name,
    createdAt: document.createdAt,
    modifiedAt: document.modifiedAt,
    canvasSize: document.canvasSize,
    layers: document.layers,
    selectedLayerId: document.selectedLayerId,
  };

  const docPath = join(docDir, DOCUMENT_JSON);
  await writeFile(docPath, JSON.stringify(onDisk, null, 2));
}

// Load assets from PNG files in assets folder
async function loadAssetsFromFolder(assetsDir: string): Promise<Asset[]> {
  const assets: Asset[] = [];

  try {
    const files = await readdir(assetsDir);
    for (const file of files) {
      if (!file.endsWith(".png")) continue;

      const filePath = join(assetsDir, file);
      const metadata = await readPngMetadata(filePath);

      if (metadata) {
        assets.push({
          filename: file,
          metadata,
        });
      }
    }
  } catch {
    // Assets folder might not exist yet
  }

  // Sort by creation time, oldest first
  assets.sort(
    (a, b) =>
      new Date(a.metadata.createdAt).getTime() -
      new Date(b.metadata.createdAt).getTime()
  );

  return assets;
}

// Load document from disk (assets loaded from PNG files)
export async function loadDocument(
  username: string,
  documentId: string
): Promise<Document | null> {
  const docDir = getDocumentDir(username, documentId);
  const docPath = join(docDir, DOCUMENT_JSON);

  try {
    const data = await readFile(docPath, "utf-8");
    const onDisk = JSON.parse(data) as DocumentOnDisk;

    // Load assets from PNG files
    const assetsDir = join(docDir, ASSETS_DIR);
    const assets = await loadAssetsFromFolder(assetsDir);

    return {
      ...onDisk,
      assets,
    };
  } catch {
    return null;
  }
}

// Delete document and all its files
export async function deleteDocument(
  username: string,
  documentId: string
): Promise<boolean> {
  const docDir = getDocumentDir(username, documentId);

  try {
    await rm(docDir, { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}

// List all documents for a user (returns summaries for efficiency)
export async function listDocuments(
  username: string
): Promise<DocumentSummary[]> {
  const docsDir = getUserDocumentsDir(username);

  try {
    const entries = await readdir(docsDir, { withFileTypes: true });
    const summaries: DocumentSummary[] = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;

      const docPath = join(docsDir, entry.name, DOCUMENT_JSON);
      try {
        const data = await readFile(docPath, "utf-8");
        const doc = JSON.parse(data) as Document;

        // Check if composite exists
        const compositePath = join(docsDir, entry.name, COMPOSITE_FILE);
        let thumbnail: string | undefined;
        try {
          await stat(compositePath);
          thumbnail = `/api/documents/${doc.id}/composite`;
        } catch {
          // No composite yet
        }

        summaries.push({
          id: doc.id,
          name: doc.name,
          modifiedAt: doc.modifiedAt,
          canvasSize: doc.canvasSize,
          layerCount: doc.layers.length,
          thumbnail,
        });
      } catch {
        // Skip invalid document directories
        continue;
      }
    }

    // Sort by modification time, newest first
    summaries.sort(
      (a, b) =>
        new Date(b.modifiedAt).getTime() - new Date(a.modifiedAt).getTime()
    );

    return summaries;
  } catch {
    // Documents directory doesn't exist yet
    return [];
  }
}

// Update specific fields of a document
export async function updateDocument(
  username: string,
  documentId: string,
  updates: {
    name?: string;
    canvasSize?: CanvasSize;
    layers?: (Layer | LayerGroup)[];
    selectedLayerId?: string | null;
  }
): Promise<Document | null> {
  const doc = await loadDocument(username, documentId);
  if (!doc) return null;

  if (updates.name !== undefined) doc.name = updates.name;
  if (updates.canvasSize) doc.canvasSize = updates.canvasSize;
  if (updates.layers) doc.layers = updates.layers;
  if (updates.selectedLayerId !== undefined)
    doc.selectedLayerId = updates.selectedLayerId;

  await saveDocument(username, doc);
  return doc;
}

// Get paths for document assets
export function getDocumentPaths(username: string, documentId: string) {
  const docDir = getDocumentDir(username, documentId);
  return {
    root: docDir,
    document: join(docDir, DOCUMENT_JSON),
    layers: join(docDir, LAYERS_DIR),
    assets: join(docDir, ASSETS_DIR),
    composite: join(docDir, COMPOSITE_FILE),
  };
}
