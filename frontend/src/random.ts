export function generateSeed(): number {
  return 1 + Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}
