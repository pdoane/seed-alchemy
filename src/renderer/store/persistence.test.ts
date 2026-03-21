import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { useAppStore } from "./appStore";
import { useImageStore, defaultImageUi } from "./imageStore";
import { useDocumentStore } from "./documentStore";
import { defaultParams } from "../../shared/defaults";

// Mock the API client
vi.mock("../api/client", () => ({
  api: {
    getState: vi.fn(),
    saveState: vi.fn(),
    getAuthSession: vi
      .fn()
      .mockResolvedValue({ authenticated: true, username: "default" }),
    getGenerationSession: vi.fn().mockResolvedValue({ generation: null }),
    login: vi.fn().mockResolvedValue(undefined),
    subscribeProgress: vi.fn().mockReturnValue(() => {}),
    getImages: vi.fn().mockResolvedValue([]),
    getDocuments: vi.fn().mockResolvedValue([]),
    getUsers: vi.fn().mockResolvedValue([]),
    getModels: vi.fn().mockResolvedValue([]),
  },
}));

import { api } from "../api/client";
import { initializeApp, switchUser } from "./persistence";

describe("persistence", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    // Reset stores to default state
    useAppStore.setState({ mode: "image" });
    useImageStore.setState({
      params: { ...defaultParams },
      images: [],
      ui: { ...defaultImageUi },
      historyIndex: -1,
      history: [],
      isGenerating: false,
      generationError: null,
      availableCheckpoints: [],
      availableLoras: [],
    });
    useDocumentStore.setState({
      documents: [],
      currentDocument: null,
      pendingTargetLayerId: null,
      documentUi: {
        currentDocumentId: null,
        activeTool: "select",
      },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("initializeApp", () => {
    it("should load state from server and apply to stores", async () => {
      const savedState = {
        mode: "gallery" as const,
        params: { ...defaultParams, prompt: "saved prompt" },
        ui: {
          selectedFilename: "saved-image.png",
          seedLocked: true,
        },
        documentUi: {
          currentDocumentId: null,
          activeTool: "select" as const,
        },
      };

      vi.mocked(api.getState).mockResolvedValue(savedState);

      const cleanup = initializeApp();

      // Wait for async initialization
      await vi.runAllTimersAsync();

      expect(api.getState).toHaveBeenCalled();
      expect(useAppStore.getState().mode).toBe("gallery");
      expect(useImageStore.getState().params.prompt).toBe("saved prompt");
      expect(useImageStore.getState().ui.selectedFilename).toBe(
        "saved-image.png"
      );
      expect(useImageStore.getState().ui.seedLocked).toBe(true);

      cleanup();
    });

    it("should handle null state (no saved state)", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);

      const cleanup = initializeApp();
      await vi.runAllTimersAsync();

      expect(useAppStore.getState().mode).toBe("image");
      expect(useImageStore.getState().params.prompt).toBe(defaultParams.prompt);

      cleanup();
    });

    it("should handle API errors gracefully", async () => {
      vi.mocked(api.getState).mockRejectedValue(new Error("Network error"));
      const consoleSpy = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      const cleanup = initializeApp();
      await vi.runAllTimersAsync();

      expect(consoleSpy).toHaveBeenCalled();
      expect(useAppStore.getState().mode).toBe("image");

      consoleSpy.mockRestore();
      cleanup();
    });

    it("should return cleanup function", () => {
      vi.mocked(api.getState).mockResolvedValue(null);

      const cleanup = initializeApp();

      expect(typeof cleanup).toBe("function");
      cleanup();
    });

    it("should save when app mode changes", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);
      vi.mocked(api.saveState).mockResolvedValue(undefined);

      const cleanup = initializeApp();

      useAppStore.getState().setMode("gallery");

      await vi.advanceTimersByTimeAsync(1000);

      expect(api.saveState).toHaveBeenCalled();

      cleanup();
    });

    it("should save when params change", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);
      vi.mocked(api.saveState).mockResolvedValue(undefined);

      const cleanup = initializeApp();

      useImageStore.getState().setParams({ prompt: "changed" });

      await vi.advanceTimersByTimeAsync(1000);

      expect(api.saveState).toHaveBeenCalled();

      cleanup();
    });

    it("should save when ui changes", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);
      vi.mocked(api.saveState).mockResolvedValue(undefined);

      const cleanup = initializeApp();

      useImageStore.getState().setUi({ selectedFilename: "new-image.png" });

      await vi.advanceTimersByTimeAsync(1000);

      expect(api.saveState).toHaveBeenCalled();

      cleanup();
    });

    it("should debounce multiple rapid saves", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);
      vi.mocked(api.saveState).mockResolvedValue(undefined);

      const cleanup = initializeApp();

      useAppStore.getState().setMode("gallery");
      useAppStore.getState().setMode("canvas");
      useAppStore.getState().setMode("image");

      await vi.advanceTimersByTimeAsync(1000);

      expect(api.saveState).toHaveBeenCalledTimes(1);

      cleanup();
    });
  });

  describe("switchUser", () => {
    it("should clear documentStore state when switching users", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);

      // Set up documentStore with previous user's data
      useDocumentStore.setState({
        documents: [
          {
            id: "doc1",
            name: "Old Doc",
            canvasSize: { width: 1024, height: 1024 },
            modifiedAt: "2024-01-01",
            layerCount: 1,
          },
        ],
        currentDocument: {
          id: "doc1",
          name: "Old Doc",
          canvasSize: { width: 1024, height: 1024 },
          createdAt: "2024-01-01",
          modifiedAt: "2024-01-01",
          assets: [],
          layers: [
            {
              id: "layer1",
              name: "Layer 1",
              visible: true,
              opacity: 100,
              blendMode: "normal",
              position: { x: 0, y: 0 },
            },
          ],
          selectedLayerId: "layer1",
        },
        pendingTargetLayerId: "layer1",
      });

      await switchUser("newuser");

      expect(useDocumentStore.getState().documents).toEqual([]);
      expect(useDocumentStore.getState().currentDocument).toBeNull();
      expect(useDocumentStore.getState().pendingTargetLayerId).toBeNull();
    });

    it("should not create duplicate progress subscriptions when switching users", async () => {
      vi.mocked(api.getState).mockResolvedValue(null);
      const unsubscribeFn = vi.fn();
      vi.mocked(api.subscribeProgress).mockReturnValue(unsubscribeFn);

      const cleanup = initializeApp();
      expect(api.subscribeProgress).toHaveBeenCalledTimes(1);

      await switchUser("user2");

      // SSE subscription should only be created once at app init
      expect(api.subscribeProgress).toHaveBeenCalledTimes(1);

      cleanup();
    });
  });
});
