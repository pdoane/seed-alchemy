import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  useImageStore,
  selectSelectedImage,
  selectCanNavigateBack,
  selectCanNavigateForward,
  defaultImageUi,
} from "./imageStore";
import { defaultParams } from "../../shared/defaults";
import type { ProgressEvent } from "../../shared/types/progress";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getImages: vi.fn().mockResolvedValue([]),
    getModels: vi.fn().mockResolvedValue([]),
    getCheckpointNames: vi.fn().mockResolvedValue([]),
    generate: vi.fn(),
    deleteImage: vi.fn().mockResolvedValue(undefined),
  },
}));

describe("imageStore", () => {
  beforeEach(() => {
    // Reset store to initial state
    useImageStore.setState({
      params: { ...defaultParams },
      images: [],
      ui: { ...defaultImageUi },
      historyIndex: -1,
      history: [],
      isGenerating: false,
      generationError: null,
      activePromptId: null,
      availableCheckpoints: [],
      availableLoras: [],
    });
  });

  describe("params", () => {
    it("should have default params initially", () => {
      const state = useImageStore.getState();
      expect(state.params).toEqual(defaultParams);
    });

    it("should update params partially", () => {
      const { setParams } = useImageStore.getState();
      setParams({ prompt: "test prompt", steps: 30 });

      const state = useImageStore.getState();
      expect(state.params.prompt).toBe("test prompt");
      expect(state.params.steps).toBe(30);
      expect(state.params.cfgScale).toBe(defaultParams.cfgScale);
    });

    it("should reset params to defaults", () => {
      const { setParams, resetParams } = useImageStore.getState();
      setParams({ prompt: "test", steps: 50, cfgScale: 15 });
      resetParams();

      expect(useImageStore.getState().params).toEqual(defaultParams);
    });
  });

  describe("images", () => {
    it("should set images", () => {
      const { setImages } = useImageStore.getState();
      setImages(["test-1.png"]);

      expect(useImageStore.getState().images).toEqual(["test-1.png"]);
    });

    it("should add image to beginning", () => {
      const { setImages, addImage } = useImageStore.getState();
      setImages(["test-1.png"]);

      addImage("test-2.png");

      const images = useImageStore.getState().images;
      expect(images).toHaveLength(2);
      expect(images[0]).toBe("test-2.png");
      expect(images[1]).toBe("test-1.png");
    });

    it("should select next image after deleting selected image", async () => {
      const { setImages, navigateTo, removeImage } = useImageStore.getState();
      setImages(["img-1.png", "img-2.png", "img-3.png"]);
      navigateTo("img-2.png");

      await removeImage("img-2.png");

      const state = useImageStore.getState();
      expect(state.images).toEqual(["img-1.png", "img-3.png"]);
      expect(state.ui.selectedFilename).toBe("img-3.png");
    });

    it("should select previous image when deleting last image", async () => {
      const { setImages, navigateTo, removeImage } = useImageStore.getState();
      setImages(["img-1.png", "img-2.png", "img-3.png"]);
      navigateTo("img-3.png");

      await removeImage("img-3.png");

      const state = useImageStore.getState();
      expect(state.images).toEqual(["img-1.png", "img-2.png"]);
      expect(state.ui.selectedFilename).toBe("img-2.png");
    });

    it("should clear selection when deleting only image", async () => {
      const { setImages, navigateTo, removeImage } = useImageStore.getState();
      setImages(["img-1.png"]);
      navigateTo("img-1.png");

      await removeImage("img-1.png");

      const state = useImageStore.getState();
      expect(state.images).toEqual([]);
      expect(state.ui.selectedFilename).toBeNull();
    });

    it("should keep selection when deleting non-selected image", async () => {
      const { setImages, navigateTo, removeImage } = useImageStore.getState();
      setImages(["img-1.png", "img-2.png", "img-3.png"]);
      navigateTo("img-1.png");

      await removeImage("img-2.png");

      const state = useImageStore.getState();
      expect(state.images).toEqual(["img-1.png", "img-3.png"]);
      expect(state.ui.selectedFilename).toBe("img-1.png");
    });
  });

  describe("selection", () => {
    const mockImages = ["img-1.png", "img-2.png"];

    it("should select image and add to history", () => {
      const { setImages, navigateTo } = useImageStore.getState();
      setImages(mockImages);
      navigateTo("img-1.png");

      const state = useImageStore.getState();
      expect(state.ui.selectedFilename).toBe("img-1.png");
      expect(state.history).toEqual(["img-1.png"]);
      expect(state.historyIndex).toBe(0);
    });

    it("should not add duplicate to history when selecting same image", () => {
      const { setImages, navigateTo } = useImageStore.getState();
      setImages(mockImages);
      navigateTo("img-1.png");
      navigateTo("img-1.png");

      const state = useImageStore.getState();
      expect(state.history).toEqual(["img-1.png"]);
    });

    it("should clear selection on null", () => {
      const { setImages, navigateTo } = useImageStore.getState();
      setImages(mockImages);
      navigateTo("img-1.png");
      navigateTo(null);

      expect(useImageStore.getState().ui.selectedFilename).toBeNull();
    });
  });

  describe("navigation history", () => {
    beforeEach(() => {
      useImageStore.setState({
        history: ["img-1.png", "img-2.png", "img-3.png"],
        historyIndex: 2,
        ui: { selectedFilename: "img-3.png", seedLocked: false },
      });
    });

    it("should navigate back", () => {
      const { navigateBack } = useImageStore.getState();
      navigateBack();

      const state = useImageStore.getState();
      expect(state.historyIndex).toBe(1);
      expect(state.ui.selectedFilename).toBe("img-2.png");
    });

    it("should navigate forward", () => {
      useImageStore.setState({
        historyIndex: 1,
        ui: { selectedFilename: "img-2.png", seedLocked: false },
      });
      const { navigateForward } = useImageStore.getState();
      navigateForward();

      const state = useImageStore.getState();
      expect(state.historyIndex).toBe(2);
      expect(state.ui.selectedFilename).toBe("img-3.png");
    });

    it("should not navigate back past beginning", () => {
      useImageStore.setState({
        historyIndex: 0,
        ui: { selectedFilename: "img-1.png", seedLocked: false },
      });
      const { navigateBack } = useImageStore.getState();
      navigateBack();

      expect(useImageStore.getState().historyIndex).toBe(0);
    });

    it("should not navigate forward past end", () => {
      const { navigateForward } = useImageStore.getState();
      navigateForward();

      expect(useImageStore.getState().historyIndex).toBe(2);
    });

    it("should truncate forward history when navigating to new image", () => {
      useImageStore.setState({
        historyIndex: 1,
        ui: { selectedFilename: "img-2.png", seedLocked: false },
      });
      const { navigateTo } = useImageStore.getState();
      navigateTo("img-4.png");

      const state = useImageStore.getState();
      expect(state.history).toEqual(["img-1.png", "img-2.png", "img-4.png"]);
      expect(state.historyIndex).toBe(2);
    });
  });

  describe("selectors", () => {
    const mockImages = ["img-1.png", "img-2.png"];

    it("selectSelectedImage should return selected image", () => {
      useImageStore.setState({
        images: mockImages,
        ui: { selectedFilename: "img-2.png", seedLocked: false },
      });

      const selected = selectSelectedImage(useImageStore.getState());
      expect(selected).toBe("img-2.png");
    });

    it("selectSelectedImage should return undefined when no selection", () => {
      useImageStore.setState({
        images: mockImages,
        ui: { selectedFilename: null, seedLocked: false },
      });

      const selected = selectSelectedImage(useImageStore.getState());
      expect(selected).toBeUndefined();
    });

    it("selectCanNavigateBack should return correct value", () => {
      useImageStore.setState({ history: ["a", "b"], historyIndex: 1 });
      expect(selectCanNavigateBack(useImageStore.getState())).toBe(true);

      useImageStore.setState({ historyIndex: 0 });
      expect(selectCanNavigateBack(useImageStore.getState())).toBe(false);
    });

    it("selectCanNavigateForward should return correct value", () => {
      useImageStore.setState({ history: ["a", "b"], historyIndex: 0 });
      expect(selectCanNavigateForward(useImageStore.getState())).toBe(true);

      useImageStore.setState({ historyIndex: 1 });
      expect(selectCanNavigateForward(useImageStore.getState())).toBe(false);
    });
  });

  describe("generation state", () => {
    it("should track generating state", () => {
      useImageStore.setState({ isGenerating: true });
      expect(useImageStore.getState().isGenerating).toBe(true);
    });

    it("should track generation error", () => {
      useImageStore.setState({ generationError: "Test error" });
      expect(useImageStore.getState().generationError).toBe("Test error");
    });

    it("should set activePromptId from generate() response", async () => {
      vi.mocked(api.generate).mockResolvedValueOnce({
        promptId: "test-prompt-123",
      });

      useImageStore.setState({
        params: { ...defaultParams, prompt: "test", model: "test.safetensors" },
      });

      await useImageStore.getState().generate();

      expect(useImageStore.getState().activePromptId).toBe("test-prompt-123");
    });
  });

  describe("handleProgressEvent", () => {
    beforeEach(() => {
      useImageStore.setState({
        isGenerating: false,
        generationStep: 0,
        generationMaxSteps: 0,
        generationNode: null,
        previewUrl: null,
        generationError: null,
        generationWidth: defaultParams.width,
        generationHeight: defaultParams.height,
        params: { ...defaultParams, width: 768, height: 512 },
      });
    });

    it("should handle execution_start event", () => {
      const event: ProgressEvent = {
        type: "execution_start",
        promptId: "prompt-1",
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(true);
      expect(state.generationStep).toBe(0);
      expect(state.generationMaxSteps).toBe(0);
      expect(state.generationNode).toBeNull();
      expect(state.previewUrl).toBeNull();
      expect(state.generationWidth).toBe(768);
      expect(state.generationHeight).toBe(512);
    });

    it("should handle executing event", () => {
      const event: ProgressEvent = {
        type: "executing",
        promptId: "prompt-1",
        node: "KSampler",
      };

      useImageStore.getState().handleProgressEvent(event);

      expect(useImageStore.getState().generationNode).toBe("KSampler");
    });

    it("should handle progress event", () => {
      const event: ProgressEvent = {
        type: "progress",
        promptId: "prompt-1",
        step: 10,
        maxSteps: 20,
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(true);
      expect(state.generationStep).toBe(10);
      expect(state.generationMaxSteps).toBe(20);
    });

    it("should handle preview event", () => {
      const event: ProgressEvent = {
        type: "preview",
        promptId: "prompt-1",
        previewUrl: "data:image/png;base64,abc123",
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(true);
      expect(state.previewUrl).toBe("data:image/png;base64,abc123");
    });

    it("should handle execution_success event for image mode", () => {
      useImageStore.setState({ isGenerating: true, images: [] });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [{ filename: "img-1.png" }, { filename: "img-2.png" }],
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(false);
      expect(state.images).toEqual(["img-2.png", "img-1.png"]);
      expect(state.ui.selectedFilename).toBe("img-1.png");
      expect(state.generationStep).toBe(0);
      expect(state.previewUrl).toBeNull();
    });

    it("should skip image handling for execution_success with documentId", () => {
      useImageStore.setState({ isGenerating: true, images: [] });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [{ filename: "img-1.png", assetId: "asset-1" }],
        documentId: "doc-1",
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(false);
      expect(state.images).toEqual([]); // Images not added - handled by documentStore
      expect(state.ui.selectedFilename).toBeNull();
    });

    it("should handle execution_interrupted event", () => {
      useImageStore.setState({
        isGenerating: true,
        generationStep: 10,
        generationMaxSteps: 20,
        previewUrl: "data:image/png;base64,abc",
      });

      const event: ProgressEvent = {
        type: "execution_interrupted",
        promptId: "prompt-1",
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(false);
      expect(state.generationStep).toBe(0);
      expect(state.generationMaxSteps).toBe(0);
      expect(state.previewUrl).toBeNull();
    });

    it("should handle execution_error event", () => {
      useImageStore.setState({ isGenerating: true });

      const event: ProgressEvent = {
        type: "execution_error",
        promptId: "prompt-1",
        error: "ComfyUI connection lost",
      };

      useImageStore.getState().handleProgressEvent(event);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(false);
      expect(state.generationError).toBe("ComfyUI connection lost");
      expect(state.generationStep).toBe(0);
      expect(state.previewUrl).toBeNull();
    });

    it("should ignore events from stale generations", () => {
      // Simulate generation A starting (activePromptId set by generate())
      useImageStore.setState({ activePromptId: "prompt-A" });

      // execution_start arrives for generation A
      const startEventA: ProgressEvent = {
        type: "execution_start",
        promptId: "prompt-A",
      };
      useImageStore.getState().handleProgressEvent(startEventA);
      expect(useImageStore.getState().isGenerating).toBe(true);

      // Generation A makes progress
      const progressEventA: ProgressEvent = {
        type: "progress",
        promptId: "prompt-A",
        step: 5,
        maxSteps: 20,
      };
      useImageStore.getState().handleProgressEvent(progressEventA);
      expect(useImageStore.getState().generationStep).toBe(5);

      // User cancels and starts generation B (activePromptId updated by generate())
      useImageStore.setState({ activePromptId: "prompt-B" });

      // execution_start arrives for generation B
      const startEventB: ProgressEvent = {
        type: "execution_start",
        promptId: "prompt-B",
      };
      useImageStore.getState().handleProgressEvent(startEventB);

      // Late event from generation A arrives - should be ignored
      const lateProgressA: ProgressEvent = {
        type: "progress",
        promptId: "prompt-A",
        step: 10,
        maxSteps: 20,
      };
      useImageStore.getState().handleProgressEvent(lateProgressA);

      // Step should still be 0 (from generation B start), not 10
      expect(useImageStore.getState().generationStep).toBe(0);

      // Late success from A should also be ignored
      const lateSuccessA: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-A",
        images: [{ filename: "stale-image.png" }],
      };
      useImageStore.getState().handleProgressEvent(lateSuccessA);

      // Should still be generating (B), and no stale images added
      expect(useImageStore.getState().isGenerating).toBe(true);
      expect(useImageStore.getState().images).toEqual([]);
    });
  });

  describe("restoreGenerationSession", () => {
    it("should restore generation session state", () => {
      const session = {
        promptId: "test-prompt",
        width: 1024,
        height: 768,
        step: 5,
        maxSteps: 20,
        previewUrl: "http://example.com/preview.png",
        node: "KSampler",
      };

      useImageStore.getState().restoreGenerationSession(session);

      const state = useImageStore.getState();
      expect(state.isGenerating).toBe(true);
      expect(state.generationStep).toBe(5);
      expect(state.generationMaxSteps).toBe(20);
      expect(state.generationNode).toBe("KSampler");
      expect(state.previewUrl).toBe("http://example.com/preview.png");
      expect(state.generationWidth).toBe(1024);
      expect(state.generationHeight).toBe(768);
    });
  });

  describe("restoreUserState", () => {
    it("should restore user state with merging defaults", () => {
      useImageStore
        .getState()
        .restoreUserState(
          { prompt: "restored prompt", steps: 25 },
          { seedLocked: true }
        );

      const state = useImageStore.getState();
      expect(state.params.prompt).toBe("restored prompt");
      expect(state.params.steps).toBe(25);
      expect(state.params.cfgScale).toBe(defaultParams.cfgScale);
      expect(state.ui.seedLocked).toBe(true);
      expect(state.ui.selectedFilename).toBeNull();
    });

    it("should handle empty partial state", () => {
      useImageStore.getState().restoreUserState({}, {});

      const state = useImageStore.getState();
      expect(state.params).toEqual(defaultParams);
      expect(state.ui).toEqual(defaultImageUi);
    });
  });

  describe("resetForUserSwitch", () => {
    it("should reset state for user switch", () => {
      // Set up some state
      useImageStore.setState({
        images: ["img1.png", "img2.png"],
        params: { ...defaultParams, prompt: "test prompt" },
        ui: { selectedFilename: "img1.png", seedLocked: true },
        historyIndex: 1,
        history: ["img1.png", "img2.png"],
        isGenerating: true,
        generationError: "some error",
        generationStep: 5,
        generationMaxSteps: 20,
        generationNode: "KSampler",
        previewUrl: "http://example.com/preview.png",
      });

      useImageStore.getState().resetForUserSwitch();

      const state = useImageStore.getState();
      expect(state.images).toEqual([]);
      expect(state.params).toEqual(defaultParams);
      expect(state.ui).toEqual(defaultImageUi);
      expect(state.historyIndex).toBe(-1);
      expect(state.history).toEqual([]);
      expect(state.isGenerating).toBe(false);
      expect(state.generationError).toBeNull();
      expect(state.generationStep).toBe(0);
      expect(state.generationMaxSteps).toBe(0);
      expect(state.generationNode).toBeNull();
      expect(state.previewUrl).toBeNull();
    });
  });

  describe("insertPromptFragment", () => {
    it("should append fragment to empty prompt", () => {
      useImageStore.setState({
        params: { ...defaultParams, prompt: "" },
      });

      useImageStore.getState().insertPromptFragment("test fragment", "prompt");

      expect(useImageStore.getState().params.prompt).toBe("test fragment");
    });

    it("should append fragment with newline to non-empty prompt", () => {
      useImageStore.setState({
        params: { ...defaultParams, prompt: "existing prompt" },
      });

      useImageStore.getState().insertPromptFragment("new fragment", "prompt");

      expect(useImageStore.getState().params.prompt).toBe(
        "existing prompt\nnew fragment"
      );
    });

    it("should append fragment to empty negative prompt", () => {
      useImageStore.setState({
        params: { ...defaultParams, negativePrompt: "" },
      });

      useImageStore
        .getState()
        .insertPromptFragment("negative fragment", "negative");

      expect(useImageStore.getState().params.negativePrompt).toBe(
        "negative fragment"
      );
    });

    it("should append fragment with newline to non-empty negative prompt", () => {
      useImageStore.setState({
        params: { ...defaultParams, negativePrompt: "existing negative" },
      });

      useImageStore.getState().insertPromptFragment("new negative", "negative");

      expect(useImageStore.getState().params.negativePrompt).toBe(
        "existing negative\nnew negative"
      );
    });
  });
});
