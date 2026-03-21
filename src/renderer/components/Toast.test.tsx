import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { Theme } from "@radix-ui/themes";
import { Toast } from "./Toast";
import { ToastContainer } from "./ToastContainer";
import { useToastStore } from "../store/toastStore";

function renderWithTheme(component: React.ReactNode) {
  return render(<Theme>{component}</Theme>);
}

describe("Toast", () => {
  beforeEach(() => {
    useToastStore.setState({ toasts: [] });
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should render toast message", () => {
    const toast = { id: "test-1", type: "info" as const, message: "Test info" };
    renderWithTheme(<Toast toast={toast} />);
    expect(screen.getByText("Test info")).toBeInTheDocument();
  });

  it("should render with alert role", () => {
    const toast = { id: "test-1", type: "info" as const, message: "Alert!" };
    renderWithTheme(<Toast toast={toast} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("should render dismiss button", () => {
    const toast = {
      id: "test-1",
      type: "success" as const,
      message: "Success!",
    };
    renderWithTheme(<Toast toast={toast} />);
    expect(screen.getByRole("button", { name: "Dismiss" })).toBeInTheDocument();
  });

  it("should remove toast when dismiss button is clicked", () => {
    useToastStore.getState().addToast({ type: "error", message: "Error!" });
    const toasts = useToastStore.getState().toasts;

    renderWithTheme(<Toast toast={toasts[0]!} />);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("should auto-dismiss after default duration", () => {
    useToastStore
      .getState()
      .addToast({ type: "info", message: "Auto dismiss" });
    const toasts = useToastStore.getState().toasts;

    renderWithTheme(<Toast toast={toasts[0]!} />);
    expect(useToastStore.getState().toasts).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(5000);
    });

    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("should auto-dismiss after custom duration", () => {
    useToastStore
      .getState()
      .addToast({ type: "info", message: "Custom duration", duration: 2000 });
    const toasts = useToastStore.getState().toasts;

    renderWithTheme(<Toast toast={toasts[0]!} />);

    act(() => {
      vi.advanceTimersByTime(1999);
    });
    expect(useToastStore.getState().toasts).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("should not auto-dismiss when duration is 0", () => {
    useToastStore
      .getState()
      .addToast({ type: "info", message: "Persistent", duration: 0 });
    const toasts = useToastStore.getState().toasts;

    renderWithTheme(<Toast toast={toasts[0]!} />);

    act(() => {
      vi.advanceTimersByTime(10000);
    });

    expect(useToastStore.getState().toasts).toHaveLength(1);
  });
});

describe("ToastContainer", () => {
  beforeEach(() => {
    useToastStore.setState({ toasts: [] });
  });

  it("should render nothing when no toasts", () => {
    renderWithTheme(<ToastContainer />);
    expect(screen.queryAllByRole("alert")).toHaveLength(0);
  });

  it("should render toasts when present", () => {
    useToastStore.getState().addToast({ type: "info", message: "Toast 1" });
    useToastStore.getState().addToast({ type: "success", message: "Toast 2" });

    renderWithTheme(<ToastContainer />);

    expect(screen.getByText("Toast 1")).toBeInTheDocument();
    expect(screen.getByText("Toast 2")).toBeInTheDocument();
  });

  it("should render multiple toasts in container", () => {
    useToastStore.getState().addToast({ type: "info", message: "First" });
    useToastStore.getState().addToast({ type: "warning", message: "Second" });
    useToastStore.getState().addToast({ type: "error", message: "Third" });

    renderWithTheme(<ToastContainer />);

    const alerts = screen.getAllByRole("alert");
    expect(alerts).toHaveLength(3);
  });
});
