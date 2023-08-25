import { useMutation, useQueryClient } from "react-query";
import { useSnapshot } from "valtio";
import { dirName } from "./util/pathUtil";
import { addImages, removeImage } from "./queries";
import { stateToImageRequest } from "./requestUtil";
import {
  CanvasImage,
  ControlNetConditionParamsState,
  GenerationParamsState,
  PromptGenResult,
  PromptGenViewState,
} from "./schema";
import { stateCanvas, stateSession, stateSettings, stateSystem } from "./store";
import { PromptGenRequest } from "./requests";
import { generateSeed } from "./random";

async function postJson(url: string, data: any) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  return response.json();
}

async function putJson(url: string, data: any) {
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data, null, 2),
  });

  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  return response.json();
}

export function usePutSettings() {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  return useMutation(
    async () => {
      return await putJson(`/api/v1/settings/${user}`, stateSettings);
    },
    {
      onSuccess: (_) => {},
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function useCancel() {
  return useMutation(
    async () => {
      return await postJson("/api/v1/cancel", {
        session_id: stateSession.sessionId,
      });
    },
    {
      onSuccess: (_) => {},
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function useGenerateImage(generation: GenerationParamsState, generatorId: string | null) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;
  const queryClient = useQueryClient();

  return useMutation(
    async () => {
      const newSeed = generation.seed.isEnabled ? generation.seed.seed : generateSeed();
      generation.seed.seed = newSeed;

      stateSession.progressAmount = -1;

      return await postJson(
        "/api/v1/sd-generate",
        stateToImageRequest(
          generation,
          stateSession.sessionId,
          generatorId,
          user,
          stateSettings.collection,
          stateSettings.safetyChecker
        )
      );
    },
    {
      onSuccess: (imagePaths: string[]) => {
        if (imagePaths.length > 0) {
          // Update React Query state
          addImages(queryClient, user, imagePaths);

          // Update selection if the images are in the current collection
          const imagePath = imagePaths[imagePaths.length - 1];
          const collection = dirName(imagePath);
          if (collection == stateSettings.collection) {
            stateSession.selectedIndex = 0;
          }

          // Add images to canvas generator
          if (generatorId) {
            const element = stateCanvas.elements.find((element) => element.id == generatorId);
            if (element) {
              for (const imagePath of imagePaths) {
                element.images.push(new CanvasImage().load({ path: imagePath }));
              }
            }
          }
        }
        stateSession.previewUrl = null;
        setTimeout(() => (stateSession.progressAmount = 0), 250);
      },
      onError: (error) => {
        console.error("Error:", error);
        stateSession.progressAmount = 0;
      },
    }
  );
}

export function useDeleteImage(imagePath: string) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;
  const queryClient = useQueryClient();

  return useMutation(
    async () => {
      return await postJson("/api/v1/image-delete", { user: user, path: imagePath });
    },
    {
      onSuccess: (_) => {
        removeImage(queryClient, user, imagePath);
      },
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function useMoveImage(srcPath: string) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;
  const queryClient = useQueryClient();

  return useMutation(
    async (dstCollection: string) => {
      return await postJson("/api/v1/image-move", { user: user, src_path: srcPath, dst_collection: dstCollection });
    },
    {
      onSuccess: (dstPath: string) => {
        removeImage(queryClient, user, srcPath);
        addImages(queryClient, user, [dstPath]);
        const collection = dirName(dstPath);
        if (collection == stateSettings.collection) {
          stateSession.selectedIndex = 0;
        }
      },
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function usePreviewProcessor(condition: ControlNetConditionParamsState) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;
  const queryClient = useQueryClient();

  return useMutation(
    async () => {
      return await postJson("/api/v1/controlnet-process", {
        user: stateSystem.user,
        collection: stateSettings.collection,
        source: condition.source,
        processor: condition.processor,
      });
    },
    {
      onSuccess: (imagePath: string) => {
        addImages(queryClient, user, [imagePath]);
        const collection = dirName(imagePath);
        if (collection == stateSettings.collection) {
          stateSession.selectedIndex = 0;
        }
      },
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function useRevealPath(path: string) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  return useMutation(
    async () => {
      return await postJson("/api/v1/reveal", { user: user, path });
    },
    {
      onSuccess: (_) => {},
      onError: (error) => {
        console.error("Error:", error);
      },
    }
  );
}

export function useGeneratePrompt(req: PromptGenRequest, stateView: PromptGenViewState) {
  return useMutation(
    async () => {
      const newSeed = stateView.seedIsEnabled ? req.seed : generateSeed();
      req.seed = newSeed;

      stateSession.progressAmount = -1;

      return await postJson("/api/v1/prompt-generate", req);
    },
    {
      onSuccess: (prompts: string[]) => {
        stateSession.promptGenResults = prompts.map((x) => new PromptGenResult().load({ prompt: x }));
        setTimeout(() => (stateSession.progressAmount = 0), 250);
      },
      onError: (error) => {
        console.error("Error:", error);
        stateSession.progressAmount = 0;
      },
    }
  );
}
