import type { FastifyInstance } from "fastify";
import { readFile, writeFile, mkdir, readdir, rm, stat } from "fs/promises";
import bcrypt from "bcrypt";
import type { User, UserCredentials } from "../../shared/types/users.js";
import {
  USERS_DIR,
  validateUsername,
  getUserDataDir,
  getUserImagesDir,
  getUserCredentialsFile,
} from "../user-utils.js";

const BCRYPT_ROUNDS = 10;

// Read user credentials from their directory
async function readUserCredentials(
  username: string
): Promise<UserCredentials | null> {
  try {
    const data = await readFile(getUserCredentialsFile(username), "utf-8");
    return JSON.parse(data) as UserCredentials;
  } catch {
    return null;
  }
}

// Write user credentials to their directory
async function writeUserCredentials(
  username: string,
  cred: UserCredentials
): Promise<void> {
  const userDir = getUserDataDir(username);
  await mkdir(userDir, { recursive: true });
  await writeFile(
    getUserCredentialsFile(username),
    JSON.stringify(cred, null, 2)
  );
}

// Scan users directory for all users
async function scanUsers(): Promise<User[]> {
  const users: User[] = [];
  try {
    const entries = await readdir(USERS_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory() && validateUsername(entry.name)) {
        const cred = await readUserCredentials(entry.name);
        users.push({
          username: entry.name,
          hasPassword: !!cred?.passwordHash,
        });
      }
    }
  } catch {
    // Return empty array if USERS_DIR doesn't exist
  }
  return users;
}

export function registerUserRoutes(fastify: FastifyInstance) {
  // List all users
  fastify.get("/api/users", async () => {
    const users = await scanUsers();
    users.sort((a, b) => a.username.localeCompare(b.username));
    return users;
  });

  // Create user
  fastify.post<{ Body: { username: string; password?: string } }>(
    "/api/users",
    async (request, reply) => {
      const { username, password } = request.body;

      // Validate username
      if (!username || !validateUsername(username)) {
        return reply.status(400).send({
          error:
            "Invalid username. Use 1-32 alphanumeric characters, underscores, or hyphens.",
        });
      }

      // Check if directory already exists
      const userDir = getUserDataDir(username);
      try {
        await stat(userDir);
        return reply.status(409).send({ error: "User already exists" });
      } catch {
        // Directory doesn't exist, good to proceed
      }

      // Create user directory structure
      const imagesDir = getUserImagesDir(username);
      await mkdir(imagesDir, { recursive: true });

      // Create user credentials if password provided
      if (password) {
        const cred: UserCredentials = {
          passwordHash: await bcrypt.hash(password, BCRYPT_ROUNDS),
        };
        await writeUserCredentials(username, cred);
      }

      return { success: true, username };
    }
  );

  // Delete user
  fastify.delete<{ Params: { username: string } }>(
    "/api/users/:username",
    async (request, reply) => {
      const { username } = request.params;

      if (!validateUsername(username)) {
        return reply.status(400).send({ error: "Invalid username" });
      }

      // Delete user directory (which contains user.json and all user data)
      const userDir = getUserDataDir(username);
      try {
        await rm(userDir, { recursive: true, force: true });
      } catch (error) {
        fastify.log.warn(
          error,
          `Failed to delete user directory for ${username}`
        );
      }

      return { success: true };
    }
  );

  // Verify password
  fastify.post<{ Params: { username: string }; Body: { password: string } }>(
    "/api/users/:username/verify",
    async (request, reply) => {
      const { username } = request.params;
      const { password } = request.body;

      if (!validateUsername(username)) {
        return reply.status(400).send({ error: "Invalid username" });
      }

      const cred = await readUserCredentials(username);

      // If no credentials file or no password, allow access
      if (!cred || !cred.passwordHash) {
        return { valid: true };
      }

      // Verify password
      const valid = await bcrypt.compare(password, cred.passwordHash);
      return { valid };
    }
  );
}
