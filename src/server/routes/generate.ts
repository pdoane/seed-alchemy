import type { FastifyInstance } from "fastify";
import { access, copyFile, readFile, unlink, writeFile } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import type { Architecture } from "../../shared/types/models.js";
import type {
  ImageParams,
  ReferenceImageConfig,
} from "../../shared/types/image.js";
import type { ControlNetConfig } from "../../shared/types/controlnet.js";
import {
  generateImgWorkflow,
  type QueuePromptResponse,
} from "../comfyui/index.js";
import {
  CLIENT_ID,
  registerPendingPrompt,
  startGenerationSession,
} from "./progress.js";
import { readSafetensorsHeader, detectArchitecture } from "../safetensors.js";
import { getUserImagesDir, getUserWorkflowFile } from "../user-utils.js";

const COMFYUI_URL = process.env.COMFYUI_URL || "http://127.0.0.1:8000";
const COMFYUI_MODELS_DIR =
  process.env.COMFYUI_MODELS_DIR ||
  join(homedir(), "Documents", "ComfyUI", "models");
const COMFYUI_INPUT_DIR =
  process.env.COMFYUI_INPUT_DIR ||
  join(homedir(), "Documents", "ComfyUI", "input");

// Model folders to search for the model file
const MODEL_FOLDERS = ["checkpoints", "diffusion_models", "unet"];

// Copy source image to ComfyUI's input folder and return the new filename
async function prepareSourceImage(
  sourceImage: string,
  username: string
): Promise<string> {
  if (!sourceImage) {
    return "";
  }

  const imagesDir = getUserImagesDir(username);
  const sourcePath = join(imagesDir, sourceImage);
  const destFilename = `seedalchemy_src_${sourceImage}`;
  const destPath = join(COMFYUI_INPUT_DIR, destFilename);

  try {
    await copyFile(sourcePath, destPath);
    return destFilename;
  } catch (error) {
    console.error(`Failed to copy source image ${sourceImage}:`, error);
    return "";
  }
}

// Copy reference images to ComfyUI's input folder and return updated configs
async function prepareReferenceImages(
  referenceImages: ReferenceImageConfig[],
  username: string
): Promise<ReferenceImageConfig[]> {
  if (!referenceImages || referenceImages.length === 0) {
    return [];
  }

  const imagesDir = getUserImagesDir(username);
  const preparedImages: ReferenceImageConfig[] = [];

  for (const ref of referenceImages) {
    const sourcePath = join(imagesDir, ref.filename);
    // Use a prefixed name to avoid conflicts and identify SeedAlchemy images
    const destFilename = `seedalchemy_ref_${ref.filename}`;
    const destPath = join(COMFYUI_INPUT_DIR, destFilename);

    try {
      await copyFile(sourcePath, destPath);
      preparedImages.push({ filename: destFilename });
    } catch (error) {
      console.error(`Failed to copy reference image ${ref.filename}:`, error);
      // Skip this reference image if copy fails
    }
  }

  return preparedImages;
}

// Copy ControlNet images to ComfyUI's input folder and return updated configs
async function prepareControlNetImages(
  controlNets: ControlNetConfig[],
  username: string
): Promise<ControlNetConfig[]> {
  if (!controlNets || controlNets.length === 0) {
    return [];
  }

  const imagesDir = getUserImagesDir(username);
  const preparedControlNets: ControlNetConfig[] = [];

  for (const config of controlNets) {
    const sourcePath = join(imagesDir, config.image);
    const destFilename = `seedalchemy_ctrl_${config.image}`;
    const destPath = join(COMFYUI_INPUT_DIR, destFilename);

    try {
      await copyFile(sourcePath, destPath);
      preparedControlNets.push({
        ...config,
        image: destFilename,
      });
    } catch (error) {
      console.error(`Failed to copy ControlNet image ${config.image}:`, error);
      // Skip this ControlNet if copy fails
    }
  }

  return preparedControlNets;
}

// Look up the architecture for a model by filename
async function getModelArchitecture(
  modelFilename: string
): Promise<Architecture | undefined> {
  // Search for the model in known folders
  for (const folder of MODEL_FOLDERS) {
    const modelPath = join(COMFYUI_MODELS_DIR, folder, modelFilename);
    try {
      await access(modelPath);
      // Found the model, read its architecture
      if (modelFilename.toLowerCase().endsWith(".safetensors")) {
        const header = await readSafetensorsHeader(modelPath);
        const { architecture } = detectArchitecture(header);
        return architecture;
      }
    } catch {
      // Model not in this folder, continue searching
    }
  }
  return undefined;
}

interface GenerateRequest {
  params: ImageParams;
  imageCount?: number;
  documentId?: string; // If set, save to document assets instead of global images
}

interface GenerateResponse {
  promptId: string;
}

export function registerGenerateRoutes(fastify: FastifyInstance) {
  // Queue a generation request (fire-and-forget)
  fastify.post<{ Body: GenerateRequest }>(
    "/api/generate",
    async (request, reply) => {
      const { params, imageCount = 1, documentId } = request.body;

      if (!params.model) {
        return reply.status(400).send({ error: "Model is required" });
      }

      if (!params.prompt) {
        return reply.status(400).send({ error: "Prompt is required" });
      }

      try {
        // Look up model architecture
        const architecture = await getModelArchitecture(params.model);

        // Prepare source image (copy to ComfyUI input folder)
        const preparedSourceImage = await prepareSourceImage(
          params.sourceImage,
          request.currentUser
        );

        // Prepare reference images (copy to ComfyUI input folder)
        const preparedReferenceImages = await prepareReferenceImages(
          params.referenceImages || [],
          request.currentUser
        );

        // Prepare ControlNet images (copy to ComfyUI input folder)
        const preparedControlNets = await prepareControlNetImages(
          params.controlNets || [],
          request.currentUser
        );

        // Create params with prepared images
        const workflowParams: ImageParams = {
          ...params,
          sourceImage: preparedSourceImage,
          referenceImages: preparedReferenceImages,
          controlNets: preparedControlNets,
        };

        // Generate workflow
        const workflow = generateImgWorkflow(
          workflowParams,
          imageCount,
          architecture
        );

        // Save workflow for debugging
        const workflowFile = getUserWorkflowFile(request.currentUser);
        await writeFile(workflowFile, JSON.stringify(workflow, null, 2));

        // Queue the prompt with client_id to link with our WebSocket connection
        const queueResponse = await fetch(`${COMFYUI_URL}/prompt`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: workflow, client_id: CLIENT_ID }),
        });

        if (!queueResponse.ok) {
          const error = await queueResponse.text();
          return reply.status(500).send({ error: `ComfyUI error: ${error}` });
        }

        const queueResult = (await queueResponse.json()) as QueuePromptResponse;
        const promptId = queueResult.prompt_id;

        // Register pending prompt for image processing on completion
        registerPendingPrompt(
          promptId,
          { type: "generate", params },
          request.currentUser,
          documentId
        );

        // Start generation session for client reconnection support
        startGenerationSession(promptId, params.width, params.height);

        return { promptId } satisfies GenerateResponse;
      } catch (error) {
        fastify.log.error(error);
        return reply.status(500).send({
          error: error instanceof Error ? error.message : "Generation failed",
        });
      }
    }
  );

  // Serve image from local storage
  fastify.get<{ Params: { filename: string } }>(
    "/api/images/:filename",
    async (request, reply) => {
      const { filename } = request.params;
      const imagesDir = getUserImagesDir(request.currentUser);

      // Sanitize filename to prevent directory traversal
      const safeName = filename.replace(/[/\\]/g, "");
      const filePath = join(imagesDir, safeName);

      try {
        const buffer = await readFile(filePath);
        const ext = safeName.split(".").pop()?.toLowerCase();
        const contentType =
          ext === "jpg" || ext === "jpeg"
            ? "image/jpeg"
            : ext === "webp"
              ? "image/webp"
              : "image/png";

        return reply.header("Content-Type", contentType).send(buffer);
      } catch {
        return reply.status(404).send({ error: "Image not found" });
      }
    }
  );

  // Delete an image
  fastify.delete<{ Params: { filename: string } }>(
    "/api/images/:filename",
    async (request, reply) => {
      const { filename } = request.params;
      const imagesDir = getUserImagesDir(request.currentUser);

      // Sanitize filename to prevent directory traversal
      const safeName = filename.replace(/[/\\]/g, "");
      const filePath = join(imagesDir, safeName);

      try {
        await unlink(filePath);
        return { success: true };
      } catch {
        return reply.status(404).send({ error: "Image not found" });
      }
    }
  );
}
