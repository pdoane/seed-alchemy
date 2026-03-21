import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { Theme } from "@radix-ui/themes";
import { ModelsMode } from "./ModelsMode";

// Mock the API client
const mockGet = vi.fn();
vi.mock("../../api/client", () => ({
  api: {
    get: (...args: unknown[]) => mockGet(...args),
  },
}));

function renderWithTheme(component: React.ReactNode) {
  return render(<Theme>{component}</Theme>);
}

describe("ModelsMode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows loading spinner initially", () => {
    mockGet.mockImplementation(() => new Promise(() => {})); // Never resolves
    renderWithTheme(<ModelsMode />);
    expect(document.querySelector(".rt-Spinner")).toBeInTheDocument();
  });

  it("shows empty state when no folders found", async () => {
    mockGet.mockResolvedValueOnce([]); // folders
    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("No model folders found")).toBeInTheDocument();
    });
  });

  it("renders folder sidebar with counts", async () => {
    mockGet
      .mockResolvedValueOnce([
        { folder: "checkpoints", count: 5 },
        { folder: "loras", count: 10 },
      ])
      .mockResolvedValueOnce([]); // models for first folder

    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("Checkpoints")).toBeInTheDocument();
      expect(screen.getByText("LoRAs")).toBeInTheDocument();
      expect(screen.getByText("5")).toBeInTheDocument();
      expect(screen.getByText("10")).toBeInTheDocument();
    });
  });

  it("shows empty models message when folder has no models", async () => {
    mockGet
      .mockResolvedValueOnce([{ folder: "checkpoints", count: 0 }])
      .mockResolvedValueOnce([]);

    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("No models in this folder")).toBeInTheDocument();
    });
  });

  it("renders model cards with info", async () => {
    mockGet
      .mockResolvedValueOnce([{ folder: "checkpoints", count: 1 }])
      .mockResolvedValueOnce([
        {
          filename: "test-model.safetensors",
          path: "checkpoints/test-model.safetensors",
          folder: "checkpoints",
          tensorCount: 1234,
          fileSizeBytes: 6_500_000_000,
          architecture: "sdxl",
          architectureSource: "modelspec",
        },
      ]);

    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("test-model.safetensors")).toBeInTheDocument();
      expect(screen.getByText("SDXL")).toBeInTheDocument();
      expect(screen.getByText("6.5 GB")).toBeInTheDocument();
    });
  });

  it("does not show architecture badge for unknown architecture", async () => {
    mockGet
      .mockResolvedValueOnce([{ folder: "checkpoints", count: 1 }])
      .mockResolvedValueOnce([
        {
          filename: "unknown-model.safetensors",
          path: "checkpoints/unknown-model.safetensors",
          folder: "checkpoints",
          tensorCount: 0,
          fileSizeBytes: 1_000_000_000,
          architecture: "unknown",
          architectureSource: "unknown",
        },
      ]);

    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("unknown-model.safetensors")).toBeInTheDocument();
      expect(screen.queryByText("UNKNOWN")).not.toBeInTheDocument();
    });
  });

  it("shows error message on API failure", async () => {
    mockGet.mockRejectedValueOnce(new Error("Connection failed"));

    renderWithTheme(<ModelsMode />);

    await waitFor(() => {
      expect(screen.getByText("Connection failed")).toBeInTheDocument();
    });
  });
});
