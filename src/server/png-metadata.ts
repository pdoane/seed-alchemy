import { readFile, writeFile } from "fs/promises";
import extractChunks from "png-chunks-extract";
import encodeChunks from "png-chunks-encode";
import text from "png-chunk-text";
import type { ImageMetadata } from "../shared/types/image.js";

export const METADATA_KEY = "SeedAlchemy";

interface PngChunk {
  name: string;
  data: Uint8Array;
}

// Read SeedAlchemy metadata from a PNG file
export async function readPngMetadata(
  filePath: string
): Promise<ImageMetadata | null> {
  try {
    const buffer = await readFile(filePath);
    const chunks = extractChunks(buffer) as PngChunk[];

    for (const chunk of chunks) {
      if (chunk.name === "tEXt" || chunk.name === "iTXt") {
        const decoded = text.decode(chunk.data);
        if (decoded.keyword === METADATA_KEY) {
          // Decode base64 then parse JSON
          const jsonString = Buffer.from(decoded.text, "base64").toString(
            "utf-8"
          );
          return JSON.parse(jsonString) as ImageMetadata;
        }
      }
    }

    return null;
  } catch {
    return null;
  }
}

// Write SeedAlchemy metadata to a PNG file
export async function writePngMetadata(
  filePath: string,
  metadata: ImageMetadata
): Promise<void> {
  const buffer = await readFile(filePath);
  const chunks = extractChunks(buffer) as PngChunk[];

  // Remove any existing SeedAlchemy metadata
  const filteredChunks = chunks.filter((chunk) => {
    if (chunk.name !== "tEXt" && chunk.name !== "iTXt") {
      return true;
    }
    try {
      const decoded = text.decode(chunk.data);
      return decoded.keyword !== METADATA_KEY;
    } catch {
      return true;
    }
  });

  // Create new metadata chunk (base64 encode JSON to handle unicode)
  const base64Data = Buffer.from(JSON.stringify(metadata)).toString("base64");
  const metadataChunk = text.encode(METADATA_KEY, base64Data);

  // Insert before IEND (last chunk)
  const iendIndex = filteredChunks.findIndex((chunk) => chunk.name === "IEND");
  if (iendIndex === -1) {
    throw new Error("Invalid PNG: no IEND chunk found");
  }

  filteredChunks.splice(iendIndex, 0, metadataChunk);

  // Encode and write
  const newBuffer = encodeChunks(filteredChunks);
  await writeFile(filePath, newBuffer);
}

// Embed metadata into a PNG buffer (for newly generated images)
export function embedMetadataInBuffer(
  buffer: Buffer,
  metadata: ImageMetadata
): Buffer {
  const chunks = extractChunks(buffer) as PngChunk[];

  // Create metadata chunk (base64 encode JSON to handle unicode)
  const base64Data = Buffer.from(JSON.stringify(metadata)).toString("base64");
  const metadataChunk = text.encode(METADATA_KEY, base64Data);

  // Insert before IEND
  const iendIndex = chunks.findIndex((chunk) => chunk.name === "IEND");
  if (iendIndex === -1) {
    throw new Error("Invalid PNG: no IEND chunk found");
  }

  chunks.splice(iendIndex, 0, metadataChunk);

  return Buffer.from(encodeChunks(chunks));
}
