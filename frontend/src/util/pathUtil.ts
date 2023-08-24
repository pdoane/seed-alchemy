export function dirName(path: string): string {
  return path.split("/").slice(0, -1).join("/");
}

export function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}
