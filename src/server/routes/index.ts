import type { FastifyInstance } from "fastify";
import { registerDocumentRoutes } from "./documents.js";
import { registerEnhanceRoutes } from "./enhance.js";
import { registerGenerateRoutes } from "./generate.js";
import { registerImageRoutes } from "./images.js";
import { registerModelRoutes } from "./models.js";
import { registerPreprocessRoutes } from "./preprocess.js";
import { registerProgressRoutes } from "./progress.js";
import { registerStateRoutes } from "./state.js";
import { registerUserRoutes } from "./users.js";
import { validateUsername } from "../user-utils.js";

// Extend FastifyRequest to include currentUser
declare module "fastify" {
  interface FastifyRequest {
    currentUser: string;
  }
}

const COOKIE_NAME = "sa_user";

// Routes that don't require authentication
const PUBLIC_ROUTES = [
  "/api/health",
  "/api/config",
  "/api/auth",
  "/api/session",
  "/api/progress", // SSE doesn't send cookies reliably
];

export function registerRoutes(fastify: FastifyInstance) {
  // Add hook to extract user from session cookie
  fastify.addHook("preHandler", async (request, reply) => {
    // Skip auth for public routes
    if (PUBLIC_ROUTES.some((route) => request.url.startsWith(route))) {
      return;
    }

    const username = request.cookies[COOKIE_NAME];

    if (!username) {
      return reply.status(401).send({ error: "Not authenticated" });
    }
    if (!validateUsername(username)) {
      return reply.status(401).send({ error: "Invalid session" });
    }
    request.currentUser = username;
  });

  fastify.get("/api/health", async () => {
    return { status: "ok", timestamp: new Date().toISOString() };
  });

  fastify.get("/api/config", async () => {
    return {
      version: "0.1.0",
      dataDir: "~/SeedAlchemy",
    };
  });

  // Set auth session cookie (login)
  fastify.post<{ Body: { username: string } }>(
    "/api/auth",
    async (request, reply) => {
      const { username } = request.body;

      if (!username || !validateUsername(username)) {
        return reply.status(400).send({ error: "Invalid username" });
      }

      reply.setCookie(COOKIE_NAME, username, {
        path: "/",
        httpOnly: true,
        sameSite: "lax",
      });

      return { success: true, username };
    }
  );

  // Clear auth session cookie (logout)
  fastify.delete("/api/auth", async (_request, reply) => {
    reply.clearCookie(COOKIE_NAME, { path: "/" });
    return { success: true };
  });

  // Get current auth session
  fastify.get("/api/auth", async (request) => {
    const username = request.cookies[COOKIE_NAME];
    if (username && validateUsername(username)) {
      return { authenticated: true, username };
    }
    return { authenticated: false };
  });

  registerUserRoutes(fastify);
  registerDocumentRoutes(fastify);
  registerEnhanceRoutes(fastify);
  registerGenerateRoutes(fastify);
  registerImageRoutes(fastify);
  registerModelRoutes(fastify);
  registerPreprocessRoutes(fastify);
  registerProgressRoutes(fastify);
  registerStateRoutes(fastify);
}
