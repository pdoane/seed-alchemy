// Mutate dst by assigning values from src, only for keys that exist in dst
/* eslint-disable @typescript-eslint/no-explicit-any */
function assign(dst: any, src: any): void {
  if (src === null || src === undefined) return;

  if (Array.isArray(dst)) {
    if (!Array.isArray(src)) return;
    for (let i = 0; i < src.length; i++) {
      if (i < dst.length && typeof dst[i] === "object" && dst[i] !== null) {
        assign(dst[i], src[i]);
      } else if (i < dst.length) {
        dst[i] = src[i];
      } else {
        dst.push(src[i]);
      }
    }
    dst.length = src.length;
  } else if (typeof dst === "object" && dst !== null) {
    if (typeof src !== "object" || src === null || Array.isArray(src)) return;
    for (const key in dst) {
      if (!(key in src)) continue;
      if (typeof dst[key] === "object" && dst[key] !== null) {
        assign(dst[key], src[key]);
      } else {
        dst[key] = src[key];
      }
    }
  }
}
/* eslint-enable @typescript-eslint/no-explicit-any */

// Merge loaded data onto a clone of defaults, filtering to only known fields
export function mergeWithDefaults<T>(defaults: T, loaded: unknown): T {
  const result = structuredClone(defaults);
  assign(result, loaded);
  return result;
}
