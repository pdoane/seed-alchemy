import { homedir } from "os";
import { join } from "path";

export const DATA_DIR = join(homedir(), "Documents/SeedAlchemy");
export const USERS_DIR = join(DATA_DIR, "users");

// Username validation: alphanumeric + underscore + hyphen, 1-32 chars
const USERNAME_REGEX = /^[a-zA-Z0-9_-]{1,32}$/;

export function validateUsername(username: string): boolean {
  return USERNAME_REGEX.test(username);
}

// Sanitize username for filesystem use (remove any unexpected chars)
export function sanitizeUsername(username: string): string {
  return username.replace(/[^a-zA-Z0-9_-]/g, "");
}

export function getUserDataDir(username: string): string {
  const safeName = sanitizeUsername(username);
  return join(USERS_DIR, safeName);
}

export function getUserImagesDir(username: string): string {
  return join(getUserDataDir(username), "images");
}

export function getUserCredentialsFile(username: string): string {
  return join(getUserDataDir(username), "credentials.json");
}

export function getUserStateFile(username: string): string {
  return join(getUserDataDir(username), "state.json");
}

export function getUserWorkflowFile(username: string): string {
  return join(getUserDataDir(username), "workflow.json");
}

export function getUserDocumentsDir(username: string): string {
  return join(getUserDataDir(username), "documents");
}

export function getDocumentDir(username: string, documentId: string): string {
  // Sanitize document ID to prevent directory traversal
  const safeId = documentId.replace(/[/\\]/g, "");
  return join(getUserDocumentsDir(username), safeId);
}
