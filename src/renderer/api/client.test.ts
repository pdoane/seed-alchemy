import { describe, it, expect, beforeEach, vi } from "vitest";
import { api } from "./client";
import type { ImageParams } from "../../shared/types/image";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("api client", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe("generate", () => {
    const params: ImageParams = {
      prompt: "test prompt",
      negativePrompt: "",
      model: "test-model.safetensors",
      sampler: "euler",
      scheduler: "normal",
      width: 512,
      height: 512,
      steps: 20,
      cfgScale: 7,
      seed: 12345,
      loras: [],
      sourceImage: "",
      sourceImageStrength: 0.75,
      referenceImages: [],
      referenceWeight: 1,
      referenceWeightType: "linear",
      referenceCombineMode: "concat",
      controlNets: [],
      faceDetailer: false,
      upscaleEnabled: false,
      upscaleFactor: 2,
    };

    it("should send generate request and return promptId", async () => {
      const mockResponse = {
        promptId: "prompt-123",
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.generate(params, 2);

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/generate",
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ params, imageCount: 2 }),
        }
      );
      expect(result).toEqual(mockResponse);
    });

    it("should default imageCount to 1", async () => {
      const mockResponse = { promptId: "prompt-123" };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      await api.generate(params);

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/generate",
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ params, imageCount: 1 }),
        }
      );
    });

    it("should throw error on failed response", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ error: "Generation failed" }),
      });

      await expect(api.generate(params)).rejects.toThrow("Generation failed");
    });
  });

  describe("getModels", () => {
    it("should fetch and return models list", async () => {
      const mockModels = [
        { filename: "model1.safetensors", folder: "checkpoints" },
        { filename: "model2.safetensors", folder: "checkpoints" },
      ];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModels),
      });

      const result = await api.getModels();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/models",
        {
          credentials: "include",
        }
      );
      expect(result).toEqual(mockModels);
    });

    it("should fetch models filtered by folder", async () => {
      const mockModels = [{ filename: "lora1.safetensors", folder: "loras" }];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModels),
      });

      const result = await api.getModels("loras");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/models?folder=loras",
        { credentials: "include" }
      );
      expect(result).toEqual(mockModels);
    });

    it("should throw error on failed response", async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });

      await expect(api.getModels()).rejects.toThrow("Failed to fetch models");
    });
  });

  describe("getCheckpointNames", () => {
    it("should return checkpoint filenames", async () => {
      const mockModels = [
        { filename: "model1.safetensors", folder: "checkpoints" },
        { filename: "model2.safetensors", folder: "checkpoints" },
      ];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModels),
      });

      const result = await api.getCheckpointNames();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/models?folder=checkpoints",
        { credentials: "include" }
      );
      expect(result).toEqual(["model1.safetensors", "model2.safetensors"]);
    });
  });

  describe("getLoraNames", () => {
    it("should return LoRA filenames", async () => {
      const mockModels = [
        { filename: "lora1.safetensors", folder: "loras" },
        { filename: "lora2.safetensors", folder: "loras" },
      ];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModels),
      });

      const result = await api.getLoraNames();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/models?folder=loras",
        { credentials: "include" }
      );
      expect(result).toEqual(["lora1.safetensors", "lora2.safetensors"]);
    });
  });

  describe("getImageUrl", () => {
    it("should construct correct image URL", () => {
      const url = api.getImageUrl("test.png");
      expect(url).toBe("http://localhost:3030/api/images/test.png");
    });

    it("should encode filename", () => {
      const url = api.getImageUrl("test image.png");
      expect(url).toBe("http://localhost:3030/api/images/test%20image.png");
    });
  });

  describe("getImages", () => {
    it("should fetch and return images list", async () => {
      const mockFilenames = ["image1.png"];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ filenames: mockFilenames }),
      });

      const result = await api.getImages();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/images",
        { credentials: "include" }
      );
      expect(result).toEqual(mockFilenames);
    });

    it("should throw error on failed response", async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });

      await expect(api.getImages()).rejects.toThrow("Failed to fetch images");
    });
  });

  describe("getImageMetadata", () => {
    it("should fetch and return image metadata", async () => {
      const mockMetadata = {
        params: {
          prompt: "test",
          negativePrompt: "",
          model: "model.safetensors",
          scheduler: "euler",
          width: 512,
          height: 512,
          steps: 20,
          cfgScale: 7,
          seed: 123,
          loras: [],
          referenceImages: [],
          faceDetailer: false,
          upscaleEnabled: false,
          upscaleFactor: 2,
        },
        createdAt: "2024-01-01T00:00:00Z",
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMetadata),
      });

      const result = await api.getImageMetadata("image1.png");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/images/image1.png/metadata",
        { credentials: "include" }
      );
      expect(result).toEqual(mockMetadata);
    });

    it("should throw error on failed response", async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });

      await expect(api.getImageMetadata("test.png")).rejects.toThrow(
        "Failed to fetch image metadata"
      );
    });
  });

  describe("deleteImage", () => {
    it("should send delete request with encoded filename", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });

      await api.deleteImage("image123.png");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/images/image123.png",
        { method: "DELETE", credentials: "include" }
      );
    });

    it("should encode special characters in filename", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });

      await api.deleteImage("file name.png");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:3030/api/images/file%20name.png",
        { method: "DELETE", credentials: "include" }
      );
    });

    it("should throw error on failed response", async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });

      await expect(api.deleteImage("test.png")).rejects.toThrow(
        "Failed to delete image"
      );
    });
  });
});
