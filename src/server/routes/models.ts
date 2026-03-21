import type { FastifyInstance } from "fastify";
import "@fastify/multipart";
import {
  readdir,
  access,
  copyFile,
  rename,
  unlink,
  mkdir,
} from "node:fs/promises";
import { join, basename, relative } from "node:path";
import { homedir } from "node:os";
import { createWriteStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import type {
  ModelInfo,
  ModelFolder,
  Architecture,
} from "../../shared/types/models.js";
import { MODEL_FOLDERS } from "../../shared/constants/models.js";
import {
  readSafetensorsHeader,
  detectArchitecture,
  getFileSize,
} from "../safetensors.js";

// ComfyUI models directory - configurable via environment
const COMFYUI_MODELS_DIR =
  process.env.COMFYUI_MODELS_DIR ||
  join(homedir(), "Documents", "ComfyUI", "models");

// File extensions we recognize as models
const MODEL_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"];

interface ModelsQuerystring {
  folder?: ModelFolder;
  architecture?: Architecture;
}

interface ImportPreviewBody {
  sourcePath: string;
}

interface ImportBody {
  sourcePath: string;
  targetFolder: ModelFolder;
  keepOriginal: boolean;
}

interface ImportPreviewResponse {
  accessible: boolean;
  filename?: string;
  fileSizeBytes?: number;
  architecture?: Architecture;
  architectureSource?: string;
  suggestedFolder?: ModelFolder;
  title?: string;
  author?: string;
  error?: string;
}

export function registerModelRoutes(fastify: FastifyInstance) {
  // GET /api/models - List all models or filter by folder/architecture
  fastify.get<{ Querystring: ModelsQuerystring }>(
    "/api/models",
    async (request) => {
      const { folder, architecture } = request.query;

      // Determine which folders to scan
      const foldersToScan = folder ? [folder] : MODEL_FOLDERS;

      const models: ModelInfo[] = [];

      for (const folderName of foldersToScan) {
        const folderPath = join(COMFYUI_MODELS_DIR, folderName);

        try {
          const folderModels = await scanModelFolder(folderPath, folderName);
          models.push(...folderModels);
        } catch (error) {
          // Folder doesn't exist or is inaccessible - skip it
          fastify.log.debug(`Skipping folder ${folderName}: ${error}`);
        }
      }

      // Filter by architecture if specified
      if (architecture) {
        return models.filter((m) => m.architecture === architecture);
      }

      return models;
    }
  );

  // GET /api/models/folders - List available model folders
  fastify.get("/api/models/folders", async () => {
    const availableFolders: { folder: ModelFolder; count: number }[] = [];

    for (const folder of MODEL_FOLDERS) {
      const folderPath = join(COMFYUI_MODELS_DIR, folder);
      try {
        const files = await readdir(folderPath);
        const modelCount = files.filter((f) =>
          MODEL_EXTENSIONS.some((ext) => f.toLowerCase().endsWith(ext))
        ).length;
        if (modelCount > 0) {
          availableFolders.push({ folder, count: modelCount });
        }
      } catch {
        // Folder doesn't exist - skip
      }
    }

    return availableFolders;
  });

  // GET /api/models/:folder/:filename - Get detailed info for a specific model
  fastify.get<{ Params: { folder: ModelFolder; filename: string } }>(
    "/api/models/:folder/:filename",
    async (request, reply) => {
      const { folder, filename } = request.params;

      if (!MODEL_FOLDERS.includes(folder)) {
        return reply.status(400).send({ error: "Invalid folder" });
      }

      const filePath = join(COMFYUI_MODELS_DIR, folder, filename);

      try {
        const modelInfo = await getModelInfo(filePath, folder);
        return modelInfo;
      } catch {
        return reply.status(404).send({ error: "Model not found" });
      }
    }
  );

  // GET /api/models/config - Get models directory configuration
  fastify.get("/api/models/config", async () => {
    return {
      modelsDir: COMFYUI_MODELS_DIR,
      folders: MODEL_FOLDERS,
    };
  });

  // POST /api/models/preview - Preview a file before importing
  fastify.post<{ Body: ImportPreviewBody }>(
    "/api/models/preview",
    async (request, reply): Promise<ImportPreviewResponse> => {
      const { sourcePath } = request.body;

      if (!sourcePath) {
        return reply.status(400).send({ error: "sourcePath is required" });
      }

      // Check if the file is accessible from the server
      try {
        await access(sourcePath);
      } catch {
        return { accessible: false, error: "File not accessible from server" };
      }

      const filename = basename(sourcePath);
      const ext = filename.toLowerCase();

      // Validate file extension
      if (!MODEL_EXTENSIONS.some((e) => ext.endsWith(e))) {
        return {
          accessible: true,
          filename,
          error: `Unsupported file type. Expected: ${MODEL_EXTENSIONS.join(", ")}`,
        };
      }

      try {
        const fileSize = await getFileSize(sourcePath);
        let architecture: Architecture = "unknown";
        let architectureSource: string = "unknown";
        let title: string | undefined;
        let author: string | undefined;

        // Parse safetensors for metadata
        if (ext.endsWith(".safetensors")) {
          try {
            const header = await readSafetensorsHeader(sourcePath);
            const archResult = detectArchitecture(header);
            architecture = archResult.architecture;
            architectureSource = archResult.source;
            title = header.metadata["modelspec.title"] || undefined;
            author = header.metadata["modelspec.author"] || undefined;
          } catch {
            // Continue with unknown architecture
          }
        }

        // Suggest folder based on architecture and file characteristics
        const suggestedFolder = suggestFolder(filename, architecture, fileSize);

        return {
          accessible: true,
          filename,
          fileSizeBytes: fileSize,
          architecture,
          architectureSource,
          suggestedFolder,
          title,
          author,
        };
      } catch (err) {
        return {
          accessible: true,
          filename,
          error: err instanceof Error ? err.message : "Failed to read file",
        };
      }
    }
  );

  // POST /api/models/import - Import a model file
  fastify.post<{ Body: ImportBody }>(
    "/api/models/import",
    async (request, reply) => {
      const { sourcePath, targetFolder, keepOriginal } = request.body;

      if (!sourcePath || !targetFolder) {
        return reply
          .status(400)
          .send({ error: "sourcePath and targetFolder are required" });
      }

      if (!MODEL_FOLDERS.includes(targetFolder)) {
        return reply.status(400).send({ error: "Invalid target folder" });
      }

      // Check if source file exists
      try {
        await access(sourcePath);
      } catch {
        return reply.status(400).send({ error: "Source file not accessible" });
      }

      const filename = basename(sourcePath);
      const targetPath = join(COMFYUI_MODELS_DIR, targetFolder, filename);

      // Check if target already exists
      try {
        await access(targetPath);
        return reply.status(409).send({
          error: "A model with this name already exists in the target folder",
        });
      } catch {
        // Good - target doesn't exist
      }

      try {
        if (keepOriginal) {
          await copyFile(sourcePath, targetPath);
        } else {
          // Try rename first (faster, works on same filesystem)
          try {
            await rename(sourcePath, targetPath);
          } catch {
            // Cross-filesystem move: copy then delete
            await copyFile(sourcePath, targetPath);
            await unlink(sourcePath);
          }
        }

        // Return the imported model info
        const modelInfo = await getModelInfo(targetPath, targetFolder);
        return modelInfo;
      } catch (err) {
        return reply.status(500).send({
          error: err instanceof Error ? err.message : "Failed to import model",
        });
      }
    }
  );

  // DELETE /api/models/:folder/:filename - Delete a model file
  fastify.delete<{ Params: { folder: ModelFolder; filename: string } }>(
    "/api/models/:folder/:filename",
    async (request, reply) => {
      const { folder, filename } = request.params;

      if (!MODEL_FOLDERS.includes(folder)) {
        return reply.status(400).send({ error: "Invalid folder" });
      }

      const filePath = join(COMFYUI_MODELS_DIR, folder, filename);

      try {
        await access(filePath);
      } catch {
        return reply.status(404).send({ error: "Model not found" });
      }

      try {
        await unlink(filePath);
        return { success: true, deleted: { folder, filename } };
      } catch (err) {
        return reply.status(500).send({
          error: err instanceof Error ? err.message : "Failed to delete model",
        });
      }
    }
  );

  // POST /api/models/upload - Upload a model file directly (for browser mode)
  fastify.post("/api/models/upload", async (request, reply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: "No file uploaded" });
    }

    const filename = data.filename;
    const ext = filename.toLowerCase();

    // Validate file extension
    if (!MODEL_EXTENSIONS.some((e) => ext.endsWith(e))) {
      return reply.status(400).send({
        error: `Unsupported file type. Expected: ${MODEL_EXTENSIONS.join(", ")}`,
      });
    }

    // Get target folder from fields (default to checkpoints)
    // Note: targetFolder field must come BEFORE file in the multipart stream
    const targetFolderField = data.fields.targetFolder;
    let folder: ModelFolder = "checkpoints";

    if (targetFolderField) {
      const field = Array.isArray(targetFolderField)
        ? targetFolderField[0]
        : targetFolderField;
      if (field && field.type === "field") {
        const fieldValue = field.value as string;
        if (fieldValue && MODEL_FOLDERS.includes(fieldValue as ModelFolder)) {
          folder = fieldValue as ModelFolder;
        }
      }
    }

    // Ensure target folder exists
    const folderPath = join(COMFYUI_MODELS_DIR, folder);
    await mkdir(folderPath, { recursive: true });

    const targetPath = join(folderPath, filename);

    // Check if target already exists
    try {
      await access(targetPath);
      return reply.status(409).send({
        error: "A model with this name already exists in the target folder",
      });
    } catch {
      // Good - target doesn't exist
    }

    try {
      // Stream file directly to disk
      await pipeline(data.file, createWriteStream(targetPath));

      // Return the imported model info
      const modelInfo = await getModelInfo(targetPath, folder);
      return modelInfo;
    } catch (err) {
      // Clean up partial file on error
      try {
        await unlink(targetPath);
      } catch {
        // Ignore cleanup errors
      }
      return reply.status(500).send({
        error: err instanceof Error ? err.message : "Failed to upload model",
      });
    }
  });
}

// Scan a model folder and return basic info for all models
async function scanModelFolder(
  folderPath: string,
  folderName: ModelFolder
): Promise<ModelInfo[]> {
  const entries = await readdir(folderPath, { withFileTypes: true });
  const models: ModelInfo[] = [];

  for (const entry of entries) {
    // Skip directories for now (could add recursive scanning later)
    if (!entry.isFile()) continue;

    // Check if it's a model file
    const ext = entry.name.toLowerCase();
    if (!MODEL_EXTENSIONS.some((e) => ext.endsWith(e))) continue;

    const filePath = join(folderPath, entry.name);

    try {
      const modelInfo = await getModelInfo(filePath, folderName);
      models.push(modelInfo);
    } catch (error) {
      // Skip files that can't be parsed
      console.warn(`Failed to parse ${entry.name}: ${error}`);
    }
  }

  return models;
}

// Derive title from filename (remove extension)
function deriveTitle(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

// Get detailed model info for a single file
async function getModelInfo(
  filePath: string,
  folder: ModelFolder
): Promise<ModelInfo> {
  const filename = basename(filePath);
  const fileSize = await getFileSize(filePath);

  // Only parse safetensors files for metadata
  if (filePath.toLowerCase().endsWith(".safetensors")) {
    try {
      const header = await readSafetensorsHeader(filePath);
      const { architecture, source } = detectArchitecture(header);

      return {
        filename,
        path: relative(COMFYUI_MODELS_DIR, filePath),
        folder,
        tensorCount: Object.keys(header.tensors).length,
        fileSizeBytes: fileSize,
        architecture,
        architectureSource: source,
        title: header.metadata["modelspec.title"] || deriveTitle(filename),
        author: header.metadata["modelspec.author"] || undefined,
        description: header.metadata["modelspec.description"] || undefined,
        thumbnail: header.metadata["modelspec.thumbnail"] || undefined,
        license: header.metadata["modelspec.license"] || undefined,
        hash: header.metadata["modelspec.hash_sha256"] || undefined,
      };
    } catch (error) {
      // Fall through to basic info if parsing fails
      console.warn(
        `Failed to parse safetensors header for ${filename}:`,
        error
      );
    }
  }

  // Basic info for non-safetensors or failed parsing
  return {
    filename,
    path: relative(COMFYUI_MODELS_DIR, filePath),
    folder,
    tensorCount: 0,
    fileSizeBytes: fileSize,
    architecture: "unknown",
    architectureSource: "unknown",
    title: deriveTitle(filename),
  };
}

// Suggest target folder based on filename patterns, architecture, and file size
function suggestFolder(
  filename: string,
  architecture: Architecture,
  fileSizeBytes: number
): ModelFolder {
  const lower = filename.toLowerCase();
  const sizeGB = fileSizeBytes / 1_000_000_000;

  // Check filename patterns first
  if (
    lower.includes("lora") ||
    lower.includes("loha") ||
    lower.includes("lokr")
  ) {
    return "loras";
  }
  if (lower.includes("vae")) {
    return "vae";
  }
  if (lower.includes("controlnet") || lower.includes("control_")) {
    return "controlnet";
  }
  if (
    lower.includes("upscale") ||
    lower.includes("esrgan") ||
    lower.includes("realesrgan")
  ) {
    return "upscale_models";
  }
  if (
    lower.includes("embed") ||
    lower.includes("textual_inversion") ||
    lower.includes("ti_")
  ) {
    return "embeddings";
  }
  if (lower.includes("clip") && !lower.includes("clip_skip")) {
    return "clip";
  }
  if (lower.includes("unet")) {
    return "unet";
  }
  if (lower.includes("diffusion_model") || lower.includes("z_image")) {
    return "diffusion_models";
  }

  // Use file size as a heuristic for checkpoints vs smaller models
  // Full checkpoints are typically 2GB+ for SD1.5, 6GB+ for SDXL
  if (sizeGB > 1.5) {
    return "checkpoints";
  }

  // LoRAs are typically under 500MB
  if (sizeGB < 0.5 && architecture !== "unknown") {
    return "loras";
  }

  // Default to checkpoints for unknown
  return "checkpoints";
}
