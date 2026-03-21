import type { FastifyInstance } from "fastify";
import { readdir, stat } from "fs/promises";
import { join } from "path";
import { readPngMetadata } from "../png-metadata.js";
import { getUserImagesDir } from "../user-utils.js";

// Scan the images directory and return list of image files sorted by modification time
async function scanImages(imagesDir: string): Promise<string[]> {
  const imageExtensions = [".png", ".jpg", ".jpeg"];
  try {
    const files = await readdir(imagesDir);
    const imageFiles = files.filter((f) => {
      const lower = f.toLowerCase();
      return imageExtensions.some((ext) => lower.endsWith(ext));
    });

    // Get modification times for sorting
    const filesWithTimes: Array<{ filename: string; mtime: number }> = [];
    for (const filename of imageFiles) {
      const filePath = join(imagesDir, filename);
      const fileStat = await stat(filePath);
      filesWithTimes.push({ filename, mtime: fileStat.mtime.getTime() });
    }

    // Sort by modification time, newest first
    filesWithTimes.sort((a, b) => b.mtime - a.mtime);

    return filesWithTimes.map((f) => f.filename);
  } catch {
    return [];
  }
}

export function registerImageRoutes(fastify: FastifyInstance) {
  // Get all images (filenames only, for fast listing)
  fastify.get("/api/images", async (request) => {
    const imagesDir = getUserImagesDir(request.currentUser);
    const filenames = await scanImages(imagesDir);
    return { filenames };
  });

  // Get metadata for a specific image
  fastify.get<{ Params: { filename: string } }>(
    "/api/images/:filename/metadata",
    async (request, reply) => {
      const { filename } = request.params;
      const imagesDir = getUserImagesDir(request.currentUser);

      // Sanitize filename to prevent directory traversal
      const safeName = filename.replace(/[/\\]/g, "");
      const filePath = join(imagesDir, safeName);

      const metadata = await readPngMetadata(filePath);
      if (!metadata) {
        return reply.status(404).send({ error: "Metadata not found" });
      }

      return metadata;
    }
  );
}
