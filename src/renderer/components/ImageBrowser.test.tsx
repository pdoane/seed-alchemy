import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Theme } from "@radix-ui/themes";
import { ImageBrowser } from "./ImageBrowser";
import { useImageStore } from "../store/imageStore";
import { defaultParams } from "../../shared/defaults";

// Mock the API client
vi.mock("../api/client", () => ({
  api: {
    getImages: vi.fn().mockResolvedValue([]),
    getModels: vi.fn().mockResolvedValue([]),
    getCheckpointNames: vi.fn().mockResolvedValue([]),
    generate: vi.fn(),
    deleteImage: vi.fn().mockResolvedValue(undefined),
    getImageUrl: vi.fn(
      (filename: string) => `http://localhost:3030/api/images/${filename}`
    ),
    getImageMetadata: vi.fn().mockResolvedValue({
      params: { prompt: "test", seed: 123 },
    }),
  },
}));

function renderWithTheme(component: React.ReactNode) {
  return render(<Theme>{component}</Theme>);
}

const mockImages = ["img-1.png", "img-2.png", "img-3.png"];

describe("ImageBrowser", () => {
  beforeEach(() => {
    useImageStore.setState({
      params: { ...defaultParams },
      images: [],
      ui: { selectedFilename: null, seedLocked: false },
      historyIndex: -1,
      history: [],
      isGenerating: false,
      generationError: null,
      availableCheckpoints: [],
      availableLoras: [],
    });
  });

  it("should show empty state message when no images", () => {
    renderWithTheme(<ImageBrowser />);
    expect(screen.getByText("No images yet")).toBeInTheDocument();
  });

  it("should render image thumbnails when images exist", async () => {
    useImageStore.setState({ images: mockImages });
    renderWithTheme(<ImageBrowser />);

    await waitFor(() => {
      const thumbnails = screen.getAllByRole("img");
      expect(thumbnails).toHaveLength(3);
    });
  });

  it("should set alt text for images", async () => {
    useImageStore.setState({ images: mockImages });
    renderWithTheme(<ImageBrowser />);

    await waitFor(() => {
      expect(
        screen.getByAltText("Generated image img-1.png")
      ).toBeInTheDocument();
      expect(
        screen.getByAltText("Generated image img-2.png")
      ).toBeInTheDocument();
      expect(
        screen.getByAltText("Generated image img-3.png")
      ).toBeInTheDocument();
    });
  });

  it("should select image when clicked", async () => {
    useImageStore.setState({ images: mockImages });
    renderWithTheme(<ImageBrowser />);

    await waitFor(() => {
      expect(
        screen.getByAltText("Generated image img-1.png")
      ).toBeInTheDocument();
    });

    const firstImage = screen.getByAltText("Generated image img-1.png");
    fireEvent.click(firstImage.parentElement!);

    expect(useImageStore.getState().ui.selectedFilename).toBe("img-1.png");
  });

  it("should update selection when clicking different image", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageBrowser />);

    await waitFor(() => {
      expect(
        screen.getByAltText("Generated image img-2.png")
      ).toBeInTheDocument();
    });

    const secondImage = screen.getByAltText("Generated image img-2.png");
    fireEvent.click(secondImage.parentElement!);

    expect(useImageStore.getState().ui.selectedFilename).toBe("img-2.png");
  });
});
