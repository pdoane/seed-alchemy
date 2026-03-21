import { describe, it, expect, beforeEach } from "vitest";
import { useAppStore } from "./appStore";

describe("appStore", () => {
  beforeEach(() => {
    useAppStore.setState({ mode: "image" });
  });

  it("should have image mode as default", () => {
    const state = useAppStore.getState();
    expect(state.mode).toBe("image");
  });

  it("should change mode", () => {
    const { setMode } = useAppStore.getState();
    setMode("canvas");
    expect(useAppStore.getState().mode).toBe("canvas");
  });

  it("should switch between all modes", () => {
    const { setMode } = useAppStore.getState();

    setMode("canvas");
    expect(useAppStore.getState().mode).toBe("canvas");

    setMode("gallery");
    expect(useAppStore.getState().mode).toBe("gallery");

    setMode("image");
    expect(useAppStore.getState().mode).toBe("image");
  });
});
