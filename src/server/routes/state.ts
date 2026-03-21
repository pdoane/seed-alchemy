import type { FastifyInstance } from "fastify";
import { readFile, writeFile, mkdir } from "fs/promises";
import { dirname } from "path";
import type { UserState } from "../../shared/types/state.js";
import { getUserStateFile } from "../user-utils.js";

export function registerStateRoutes(fastify: FastifyInstance) {
  // Get persisted state
  fastify.get("/api/state", async (request, reply) => {
    const stateFile = getUserStateFile(request.currentUser);
    try {
      const data = await readFile(stateFile, "utf-8");
      return JSON.parse(data) as UserState;
    } catch {
      // Return null if no state file exists
      return reply.status(404).send({ error: "No saved state" });
    }
  });

  // Save state
  fastify.put<{ Body: UserState }>("/api/state", async (request, reply) => {
    const state = request.body;
    const stateFile = getUserStateFile(request.currentUser);

    try {
      await mkdir(dirname(stateFile), { recursive: true });
      await writeFile(stateFile, JSON.stringify(state, null, 2));
      return { success: true };
    } catch (error) {
      fastify.log.error(error);
      return reply.status(500).send({ error: "Failed to save state" });
    }
  });
}
