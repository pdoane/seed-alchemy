import type { FastifyInstance } from "fastify";
import { copyFile, writeFile } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import type { QueuePromptResponse, ComfyWorkflow } from "../comfyui/index.js";
import { CLIENT_ID, registerPendingPrompt } from "./progress.js";
import { getUserImagesDir, getUserWorkflowFile } from "../user-utils.js";
import type { DetectorParams } from "../../shared/types/image.js";

const COMFYUI_URL = process.env.COMFYUI_URL || "http://127.0.0.1:8000";
const COMFYUI_INPUT_DIR =
  process.env.COMFYUI_INPUT_DIR ||
  join(homedir(), "Documents", "ComfyUI", "input");

interface PreprocessRequest {
  image: string;
  preprocessor: string;
  resolution: number;
}

interface PreprocessResponse {
  promptId: string;
}

// Generate a preprocessing-only workflow
function generatePreprocessWorkflow(
  inputImage: string,
  preprocessor: string,
  resolution: number
): ComfyWorkflow {
  return {
    load_image: {
      class_type: "LoadImage",
      inputs: {
        image: inputImage,
      },
    },
    preprocess: {
      class_type: "AIO_Preprocessor",
      inputs: {
        image: ["load_image", 0],
        preprocessor: preprocessor,
        resolution: resolution,
      },
    },
    save: {
      class_type: "SaveImage",
      inputs: {
        images: ["preprocess", 0],
        filename_prefix: "SeedAlchemy",
      },
    },
  };
}

export function registerPreprocessRoutes(fastify: FastifyInstance) {
  // Preprocess an image and save the result
  fastify.post<{ Body: PreprocessRequest }>(
    "/api/preprocess",
    async (request, reply) => {
      const { image, preprocessor, resolution } = request.body;

      if (!image) {
        return reply.status(400).send({ error: "Image is required" });
      }

      if (!preprocessor) {
        return reply.status(400).send({ error: "Preprocessor is required" });
      }

      try {
        // Copy image to ComfyUI input folder
        const imagesDir = getUserImagesDir(request.currentUser);
        const sourcePath = join(imagesDir, image);
        const destFilename = `seedalchemy_preprocess_${image}`;
        const destPath = join(COMFYUI_INPUT_DIR, destFilename);

        await copyFile(sourcePath, destPath);

        // Generate preprocess workflow
        const workflow = generatePreprocessWorkflow(
          destFilename,
          preprocessor,
          resolution
        );

        // Save workflow for debugging
        const workflowFile = getUserWorkflowFile(request.currentUser);
        await writeFile(workflowFile, JSON.stringify(workflow, null, 2));

        // Queue the prompt
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
        const detectParams: DetectorParams = {
          source: image,
          detector: preprocessor,
          resolution,
        };
        registerPendingPrompt(
          promptId,
          { type: "detect", params: detectParams },
          request.currentUser
        );

        return { promptId } satisfies PreprocessResponse;
      } catch (error) {
        fastify.log.error(error);
        return reply.status(500).send({
          error:
            error instanceof Error ? error.message : "Preprocessing failed",
        });
      }
    }
  );
}
