import type { FastifyInstance } from "fastify";
import { access, copyFile, writeFile } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import type { Architecture } from "../../shared/types/models.js";
import type {
  EnhanceParams,
  OperationRecord,
  SingleOperationRecord,
} from "../../shared/types/image.js";
import {
  generateEnhanceWorkflow,
  type QueuePromptResponse,
} from "../comfyui/index.js";
import {
  CLIENT_ID,
  registerPendingPrompt,
  startGenerationSession,
} from "./progress.js";
import { readSafetensorsHeader, detectArchitecture } from "../safetensors.js";
import { getUserImagesDir, getUserWorkflowFile } from "../user-utils.js";
import { readPngMetadata } from "../png-metadata.js";

const COMFYUI_URL = process.env.COMFYUI_URL || "http://127.0.0.1:8188";
const COMFYUI_MODELS_DIR =
  process.env.COMFYUI_MODELS_DIR ||
  join(homedir(), "Documents", "ComfyUI", "models");
const COMFYUI_INPUT_DIR =
  process.env.COMFYUI_INPUT_DIR ||
  join(homedir(), "Documents", "ComfyUI", "input");

const MODEL_FOLDERS = ["checkpoints", "diffusion_models"];

// Copy source image to ComfyUI's input folder and return the new filename
async function prepareSourceImage(
  sourceImage: string,
  username: string
): Promise<string> {
  const imagesDir = getUserImagesDir(username);
  const sourcePath = join(imagesDir, sourceImage);
  const destFilename = `seedalchemy_enhance_${sourceImage}`;
  const destPath = join(COMFYUI_INPUT_DIR, destFilename);

  await copyFile(sourcePath, destPath);
  return destFilename;
}

// Look up the architecture for a model by filename
async function getModelArchitecture(
  modelFilename: string
): Promise<Architecture | undefined> {
  for (const folder of MODEL_FOLDERS) {
    const modelPath = join(COMFYUI_MODELS_DIR, folder, modelFilename);
    try {
      await access(modelPath);
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

// Build operation record, creating a sequence if source has existing operations
function buildOperationRecord(
  sourceMetadata: { operation: OperationRecord } | null,
  enhanceParams: EnhanceParams
): OperationRecord {
  const enhanceOp: SingleOperationRecord = {
    type: "enhance",
    params: enhanceParams,
  };

  if (!sourceMetadata) {
    return enhanceOp;
  }

  const sourceOp = sourceMetadata.operation;

  // If source is already a sequence, append to it
  if (sourceOp.type === "sequence") {
    return {
      type: "sequence",
      operations: [...sourceOp.operations, enhanceOp],
    };
  }

  // Create a new sequence with the source operation and enhance operation
  return {
    type: "sequence",
    operations: [sourceOp as SingleOperationRecord, enhanceOp],
  };
}

interface EnhanceRequest {
  params: EnhanceParams;
  documentId?: string;
}

interface EnhanceResponse {
  promptId: string;
}

export function registerEnhanceRoutes(fastify: FastifyInstance) {
  fastify.post<{ Body: EnhanceRequest }>(
    "/api/enhance",
    async (request, reply) => {
      const { params, documentId } = request.body;

      if (!params.source) {
        return reply.status(400).send({ error: "Source image is required" });
      }

      if (!params.model) {
        return reply.status(400).send({ error: "Model is required" });
      }

      try {
        // Get source image path
        const imagesDir = getUserImagesDir(request.currentUser);
        const sourcePath = join(imagesDir, params.source);

        // Read source image metadata for provenance tracking
        const sourceMetadata = await readPngMetadata(sourcePath);

        // Look up model architecture
        const architecture = await getModelArchitecture(params.model);

        // Prepare source image (copy to ComfyUI input folder)
        const preparedSourceImage = await prepareSourceImage(
          params.source,
          request.currentUser
        );

        // Generate enhance workflow
        const workflow = generateEnhanceWorkflow(
          params,
          preparedSourceImage,
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

        // Build operation record with provenance chain
        const operation = buildOperationRecord(sourceMetadata, params);

        // Register pending prompt for image processing on completion
        registerPendingPrompt(
          promptId,
          operation,
          request.currentUser,
          documentId
        );

        // Start generation session for client reconnection support
        // For enhance, we don't know the output dimensions until processing
        // Use 0,0 as placeholder - the actual dimensions will be in the output
        startGenerationSession(promptId, 0, 0);

        return { promptId } satisfies EnhanceResponse;
      } catch (error) {
        fastify.log.error(error);
        return reply.status(500).send({
          error: error instanceof Error ? error.message : "Enhance failed",
        });
      }
    }
  );
}
