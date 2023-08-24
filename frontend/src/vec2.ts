import { loadProps } from "./util/loadUtil";

export class Vec2 {
  x: number = 0.0;
  y: number = 0.0;

  static create(x: number, y: number): Vec2 {
    const v = new Vec2();
    v.x = x;
    v.y = y;
    return v;
  }

  load(src: Partial<Vec2>) {
    loadProps(this, src);
    return this;
  }
}
