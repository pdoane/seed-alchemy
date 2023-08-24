export interface Loadable {
  load(src: Partial<Loadable>): Loadable;
}

export function loadProps<T>(dst: T, src: Partial<T>): void {
  for (const key in dst) {
    const value = src[key];
    if ((typeof value !== "object" || value === null) && value !== undefined) {
      // TODO additional type checking - also null values lose their type so may not be able to use this
      dst[key] = value;
    }
  }
}

export function loadArray<T extends Loadable>(dst: T[], src: Partial<T>[] | undefined, type: new () => T): void {
  const isArray = Array.isArray(src);
  if (isArray) {
    for (const srcElem of src) {
      const dstElem = new type();
      dstElem.load(srcElem);
      dst.push(dstElem);
    }
  }
}

export function loadNumberArray(dst: number[], src: number[] | undefined): void {
  if (src !== undefined) {
    dst.splice(0, dst.length, ...src);
  }
}

export function loadOptional<T extends Loadable>(src: Partial<T> | null | undefined, type: new () => T): T | null {
  if (src === undefined || src === null) return null;
  const dst = new type();
  dst.load(src);
  return dst;
}

export function loadNew<T extends Loadable>(src: Partial<T> | null | undefined, type: new () => T): T {
  const dst = new type();
  if (src !== undefined && src !== null) dst.load(src);
  return dst;
}
