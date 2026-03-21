import { describe, it, expect, beforeEach } from "vitest";
import { useToastStore, toast } from "./toastStore";

describe("toastStore", () => {
  beforeEach(() => {
    useToastStore.setState({ toasts: [] });
  });

  it("should start with empty toasts", () => {
    const state = useToastStore.getState();
    expect(state.toasts).toEqual([]);
  });

  it("should add a toast", () => {
    const { addToast } = useToastStore.getState();
    const id = addToast({ type: "info", message: "Test message" });

    const state = useToastStore.getState();
    expect(state.toasts).toHaveLength(1);
    expect(state.toasts[0]).toEqual({
      id,
      type: "info",
      message: "Test message",
    });
  });

  it("should add toast with duration", () => {
    const { addToast } = useToastStore.getState();
    addToast({ type: "success", message: "Success!", duration: 3000 });

    const state = useToastStore.getState();
    expect(state.toasts[0]?.duration).toBe(3000);
  });

  it("should remove a toast by id", () => {
    const { addToast, removeToast } = useToastStore.getState();
    const id = addToast({ type: "error", message: "Error!" });

    expect(useToastStore.getState().toasts).toHaveLength(1);

    removeToast(id);
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("should clear all toasts", () => {
    const { addToast, clearToasts } = useToastStore.getState();
    addToast({ type: "info", message: "One" });
    addToast({ type: "warning", message: "Two" });
    addToast({ type: "error", message: "Three" });

    expect(useToastStore.getState().toasts).toHaveLength(3);

    clearToasts();
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("should generate unique ids for each toast", () => {
    const { addToast } = useToastStore.getState();
    const id1 = addToast({ type: "info", message: "First" });
    const id2 = addToast({ type: "info", message: "Second" });

    expect(id1).not.toBe(id2);
  });

  describe("toast helper functions", () => {
    it("should add info toast", () => {
      toast.info("Info message");
      const state = useToastStore.getState();
      expect(state.toasts[0]?.type).toBe("info");
      expect(state.toasts[0]?.message).toBe("Info message");
    });

    it("should add success toast", () => {
      toast.success("Success message");
      const state = useToastStore.getState();
      expect(state.toasts[0]?.type).toBe("success");
      expect(state.toasts[0]?.message).toBe("Success message");
    });

    it("should add warning toast", () => {
      toast.warning("Warning message");
      const state = useToastStore.getState();
      expect(state.toasts[0]?.type).toBe("warning");
      expect(state.toasts[0]?.message).toBe("Warning message");
    });

    it("should add error toast", () => {
      toast.error("Error message");
      const state = useToastStore.getState();
      expect(state.toasts[0]?.type).toBe("error");
      expect(state.toasts[0]?.message).toBe("Error message");
    });

    it("should support duration in helper functions", () => {
      toast.info("Message", 10000);
      const state = useToastStore.getState();
      expect(state.toasts[0]?.duration).toBe(10000);
    });
  });
});
