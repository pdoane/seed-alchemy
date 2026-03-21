import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { writeFile, mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import {
  readPngMetadata,
  writePngMetadata,
  embedMetadataInBuffer,
  METADATA_KEY,
} from "./png-metadata";
import type { ImageMetadata, ImageParams } from "../shared/types/image";

// Valid 1x1 red PNG created with proper CRC
// Generated using: pngtopam test.png | pamtopng > minimal.png
const MINIMAL_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
  "base64"
);

const testParams: ImageParams = {
  prompt: "test prompt",
  negativePrompt: "ugly",
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

const testMetadata: ImageMetadata = {
  createdAt: "2024-01-01T00:00:00Z",
  operation: { type: "generate", params: testParams },
};

describe("png-metadata", () => {
  let testDir: string;
  let testFilePath: string;

  beforeEach(async () => {
    testDir = join(tmpdir(), `png-metadata-test-${Date.now()}`);
    await mkdir(testDir, { recursive: true });
    testFilePath = join(testDir, "test.png");
  });

  afterEach(async () => {
    try {
      await rm(testDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe("METADATA_KEY", () => {
    it("should be SeedAlchemy", () => {
      expect(METADATA_KEY).toBe("SeedAlchemy");
    });
  });

  describe("writePngMetadata and readPngMetadata", () => {
    it("should write and read metadata from a PNG file", async () => {
      // Create a test PNG file
      await writeFile(testFilePath, MINIMAL_PNG);

      // Write metadata
      await writePngMetadata(testFilePath, testMetadata);

      // Read it back
      const result = await readPngMetadata(testFilePath);

      expect(result).not.toBeNull();
      expect(result?.operation).toEqual(testMetadata.operation);
      expect(result?.createdAt).toBe("2024-01-01T00:00:00Z");
    });

    it("should overwrite existing metadata", async () => {
      await writeFile(testFilePath, MINIMAL_PNG);

      // Write first metadata
      await writePngMetadata(testFilePath, testMetadata);

      // Write new metadata
      const newMetadata: ImageMetadata = {
        createdAt: "2024-06-15T12:00:00Z",
        operation: {
          type: "generate",
          params: { ...testParams, prompt: "new prompt", seed: 99999 },
        },
      };
      await writePngMetadata(testFilePath, newMetadata);

      // Read back
      const result = await readPngMetadata(testFilePath);

      expect(result?.operation.type).toBe("generate");
      const params = (result?.operation as { params: ImageParams }).params;
      expect(params.prompt).toBe("new prompt");
      expect(params.seed).toBe(99999);
      expect(result?.createdAt).toBe("2024-06-15T12:00:00Z");
    });

    it("should return null for PNG without metadata", async () => {
      await writeFile(testFilePath, MINIMAL_PNG);

      const result = await readPngMetadata(testFilePath);

      expect(result).toBeNull();
    });

    it("should return null for non-existent file", async () => {
      const result = await readPngMetadata("/non/existent/file.png");

      expect(result).toBeNull();
    });
  });

  describe("embedMetadataInBuffer", () => {
    it("should embed metadata into a PNG buffer", () => {
      const result = embedMetadataInBuffer(MINIMAL_PNG, testMetadata);

      // Result should be larger than original (metadata added)
      expect(result.length).toBeGreaterThan(MINIMAL_PNG.length);

      // Should still be a valid PNG (starts with PNG signature)
      expect(result.subarray(0, 8)).toEqual(MINIMAL_PNG.subarray(0, 8));
    });

    it("should produce a buffer that can be read back", async () => {
      const embeddedBuffer = embedMetadataInBuffer(MINIMAL_PNG, testMetadata);

      // Write to file and read back
      await writeFile(testFilePath, embeddedBuffer);
      const result = await readPngMetadata(testFilePath);

      expect(result).not.toBeNull();
      expect(result?.operation).toEqual(testMetadata.operation);
      expect(result?.createdAt).toBe("2024-01-01T00:00:00Z");
    });
  });

  describe("metadata content", () => {
    it("should preserve all ImageParams fields", async () => {
      await writeFile(testFilePath, MINIMAL_PNG);
      await writePngMetadata(testFilePath, testMetadata);

      const result = await readPngMetadata(testFilePath);

      expect(result?.operation.type).toBe("generate");
      const params = (result?.operation as { params: ImageParams }).params;
      expect(params.prompt).toBe("test prompt");
      expect(params.negativePrompt).toBe("ugly");
      expect(params.model).toBe("test-model.safetensors");
      expect(params.sampler).toBe("euler");
      expect(params.scheduler).toBe("normal");
      expect(params.width).toBe(512);
      expect(params.height).toBe(512);
      expect(params.steps).toBe(20);
      expect(params.cfgScale).toBe(7);
      expect(params.seed).toBe(12345);
    });

    it("should handle special characters in prompt", async () => {
      const specialMetadata: ImageMetadata = {
        createdAt: "2024-01-01T00:00:00Z",
        operation: {
          type: "generate",
          params: {
            ...testParams,
            prompt: "test \"quotes\" and 'apostrophes' and\nnewlines",
          },
        },
      };

      await writeFile(testFilePath, MINIMAL_PNG);
      await writePngMetadata(testFilePath, specialMetadata);

      const result = await readPngMetadata(testFilePath);

      const params = (result?.operation as { params: ImageParams }).params;
      expect(params.prompt).toBe(
        "test \"quotes\" and 'apostrophes' and\nnewlines"
      );
    });

    it("should handle unicode characters", async () => {
      const unicodeMetadata: ImageMetadata = {
        createdAt: "2024-01-01T00:00:00Z",
        operation: {
          type: "generate",
          params: {
            ...testParams,
            prompt: "日本語テスト 🎨 émojis",
          },
        },
      };

      await writeFile(testFilePath, MINIMAL_PNG);
      await writePngMetadata(testFilePath, unicodeMetadata);

      const result = await readPngMetadata(testFilePath);

      const params = (result?.operation as { params: ImageParams }).params;
      expect(params.prompt).toBe("日本語テスト 🎨 émojis");
    });
  });
});
