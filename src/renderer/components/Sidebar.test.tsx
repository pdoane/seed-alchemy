import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Theme } from "@radix-ui/themes";
import { Sidebar } from "./Sidebar";
import { useAppStore } from "../store/appStore";

function renderWithTheme(component: React.ReactNode) {
  return render(<Theme>{component}</Theme>);
}

describe("Sidebar", () => {
  beforeEach(() => {
    useAppStore.setState({ mode: "image" });
  });

  it("should render mode buttons", () => {
    renderWithTheme(<Sidebar />);
    expect(screen.getByRole("button", { name: "Image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Canvas" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Gallery" })).toBeInTheDocument();
  });

  it("should change mode when clicking buttons", () => {
    renderWithTheme(<Sidebar />);

    fireEvent.click(screen.getByRole("button", { name: "Canvas" }));
    expect(useAppStore.getState().mode).toBe("canvas");

    fireEvent.click(screen.getByRole("button", { name: "Gallery" }));
    expect(useAppStore.getState().mode).toBe("gallery");
  });
});
