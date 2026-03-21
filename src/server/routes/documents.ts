import type { FastifyInstance } from "fastify";
import { createReadStream } from "fs";
import { stat } from "fs/promises";
import type {
  CreateDocumentRequest,
  UpdateDocumentRequest,
} from "../../shared/types/document.js";
import {
  createDocument,
  deleteDocument,
  getDocumentPaths,
  listDocuments,
  loadDocument,
  saveDocument,
  updateDocument,
} from "../document-storage.js";

export function registerDocumentRoutes(fastify: FastifyInstance) {
  // List all documents
  fastify.get("/api/documents", async (request) => {
    const documents = await listDocuments(request.currentUser);
    return { documents };
  });

  // Get a single document
  fastify.get<{ Params: { id: string } }>(
    "/api/documents/:id",
    async (request, reply) => {
      const { id } = request.params;
      const doc = await loadDocument(request.currentUser, id);

      if (!doc) {
        return reply.status(404).send({ error: "Document not found" });
      }

      return doc;
    }
  );

  // Create a new document
  fastify.post<{ Body: CreateDocumentRequest }>(
    "/api/documents",
    async (request) => {
      const { name, canvasSize } = request.body;
      const doc = createDocument(name, canvasSize);
      await saveDocument(request.currentUser, doc);
      return doc;
    }
  );

  // Update a document
  fastify.put<{ Params: { id: string }; Body: UpdateDocumentRequest }>(
    "/api/documents/:id",
    async (request, reply) => {
      const { id } = request.params;
      const updates = request.body;

      const doc = await updateDocument(request.currentUser, id, updates);
      if (!doc) {
        return reply.status(404).send({ error: "Document not found" });
      }

      return doc;
    }
  );

  // Delete a document
  fastify.delete<{ Params: { id: string } }>(
    "/api/documents/:id",
    async (request, reply) => {
      const { id } = request.params;
      const success = await deleteDocument(request.currentUser, id);

      if (!success) {
        return reply.status(404).send({ error: "Document not found" });
      }

      return { success: true };
    }
  );

  // Get document composite image
  fastify.get<{ Params: { id: string } }>(
    "/api/documents/:id/composite",
    async (request, reply) => {
      const { id } = request.params;
      const paths = getDocumentPaths(request.currentUser, id);

      try {
        await stat(paths.composite);
        reply.type("image/png");
        return createReadStream(paths.composite);
      } catch {
        return reply.status(404).send({ error: "Composite not found" });
      }
    }
  );

  // Get a layer image
  fastify.get<{ Params: { id: string; filename: string } }>(
    "/api/documents/:id/layers/:filename",
    async (request, reply) => {
      const { id, filename } = request.params;
      const paths = getDocumentPaths(request.currentUser, id);

      // Sanitize filename
      const safeName = filename.replace(/[/\\]/g, "");
      const filePath = `${paths.layers}/${safeName}`;

      try {
        await stat(filePath);
        reply.type("image/png");
        return createReadStream(filePath);
      } catch {
        return reply.status(404).send({ error: "Layer image not found" });
      }
    }
  );

  // Get an asset image
  fastify.get<{ Params: { id: string; filename: string } }>(
    "/api/documents/:id/assets/:filename",
    async (request, reply) => {
      const { id, filename } = request.params;
      const paths = getDocumentPaths(request.currentUser, id);

      // Sanitize filename
      const safeName = filename.replace(/[/\\]/g, "");
      const filePath = `${paths.assets}/${safeName}`;

      try {
        await stat(filePath);
        reply.type("image/png");
        return createReadStream(filePath);
      } catch {
        return reply.status(404).send({ error: "Asset not found" });
      }
    }
  );
}
