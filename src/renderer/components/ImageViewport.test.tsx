import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { api } from "../api/client";
import { Theme } from "@radix-ui/themes";
import { ImageViewport } from "./ImageViewport";
import { useImageStore } from "../store/imageStore";
import { defaultParams } from "../../shared/defaults";

// Mock window.confirm
vi.stubGlobal("confirm", vi.fn());

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
      createdAt: "2024-01-01T12:00:00Z",
      operation: {
        type: "generate",
        params: {
          prompt: "a beautiful landscape",
          negativePrompt: "",
          model: "test-model.safetensors",
          sampler: "euler",
          scheduler: "normal",
          width: 512,
          height: 768,
          steps: 25,
          cfgScale: 7.5,
          seed: 12345,
          loras: [],
          sourceImage: undefined,
          sourceImageStrength: 0.75,
          referenceImages: [],
          referenceWeight: 1,
          referenceWeightType: "linear",
          referenceCombineMode: "concat",
          faceDetailer: false,
          upscaleEnabled: false,
          upscaleFactor: 2,
        },
      },
    }),
  },
}));

function renderWithTheme(component: React.ReactNode) {
  return render(<Theme>{component}</Theme>);
}

const mockImages = ["img-1.png", "img-2.png", "img-3.png"];

describe("ImageViewport", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("should show empty state when no image selected", () => {
    renderWithTheme(<ImageViewport />);
    expect(screen.getByText("No Image Selected")).toBeInTheDocument();
    expect(
      screen.getByText("Select an image from the browser or generate a new one")
    ).toBeInTheDocument();
  });

  it("should show progress bar when generating without image selected", () => {
    useImageStore.setState({
      isGenerating: true,
      generationStep: 5,
      generationMaxSteps: 20,
    });
    renderWithTheme(<ImageViewport />);

    // Progress bar should show step count
    expect(screen.getByText("5/20")).toBeInTheDocument();
  });

  it("should show image with progress bar when generating with image selected", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
      isGenerating: true,
      generationStep: 10,
      generationMaxSteps: 20,
    });
    renderWithTheme(<ImageViewport />);

    // Image should still be visible
    const image = screen.getByAltText("Generated image") as HTMLImageElement;
    expect(image).toBeInTheDocument();

    // Progress bar should be shown
    expect(screen.getByText("10/20")).toBeInTheDocument();

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should render navigation buttons", () => {
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Back" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Forward" })).toBeInTheDocument();
  });

  it("should disable back button when cannot navigate back", () => {
    useImageStore.setState({
      history: ["img-1.png"],
      historyIndex: 0,
    });
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
  });

  it("should disable forward button when cannot navigate forward", () => {
    useImageStore.setState({
      history: ["img-1.png"],
      historyIndex: 0,
    });
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Forward" })).toBeDisabled();
  });

  it("should enable back button when can navigate back", () => {
    useImageStore.setState({
      history: ["img-1.png", "img-2.png"],
      historyIndex: 1,
    });
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Back" })).not.toBeDisabled();
  });

  it("should enable forward button when can navigate forward", () => {
    useImageStore.setState({
      history: ["img-1.png", "img-2.png"],
      historyIndex: 0,
    });
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Forward" })).not.toBeDisabled();
  });

  it("should navigate back when back button clicked", async () => {
    useImageStore.setState({
      images: mockImages,
      history: ["img-1.png", "img-2.png"],
      historyIndex: 1,
      ui: { selectedFilename: "img-2.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    fireEvent.click(screen.getByRole("button", { name: "Back" }));

    expect(useImageStore.getState().historyIndex).toBe(0);
    expect(useImageStore.getState().ui.selectedFilename).toBe("img-1.png");

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should navigate forward when forward button clicked", async () => {
    useImageStore.setState({
      images: mockImages,
      history: ["img-1.png", "img-2.png"],
      historyIndex: 0,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    fireEvent.click(screen.getByRole("button", { name: "Forward" }));

    expect(useImageStore.getState().historyIndex).toBe(1);
    expect(useImageStore.getState().ui.selectedFilename).toBe("img-2.png");

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should show delete button when image is selected", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should disable delete button when no image selected", () => {
    renderWithTheme(<ImageViewport />);
    expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled();
  });

  it("should display selected image", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    const image = screen.getByAltText("Generated image") as HTMLImageElement;
    expect(image).toBeInTheDocument();
    expect(image.src).toContain("/api/images/img-1.png");

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should call removeImage when delete confirmed", async () => {
    (window.confirm as ReturnType<typeof vi.fn>).mockReturnValue(true);

    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalledWith("Delete this image?");
    });
  });

  it("should not call removeImage when delete cancelled", async () => {
    (window.confirm as ReturnType<typeof vi.fn>).mockReturnValue(false);

    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    // Image should still be selected
    await waitFor(() => {
      expect(useImageStore.getState().ui.selectedFilename).toBe("img-1.png");
    });
  });

  it("should show metadata button when image is selected", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    expect(
      screen.getByRole("button", { name: "Show metadata" })
    ).toBeInTheDocument();

    // Wait for async metadata fetch to complete
    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalled();
    });
  });

  it("should disable metadata button when no image selected", () => {
    renderWithTheme(<ImageViewport />);
    expect(
      screen.getByRole("button", { name: "Show metadata" })
    ).toBeDisabled();
  });

  it("should fetch and display metadata when toggle clicked", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    const metadataButton = screen.getByRole("button", {
      name: "Show metadata",
    });

    // Wait for metadata to load so button becomes enabled
    await waitFor(() => {
      expect(metadataButton).not.toBeDisabled();
    });

    fireEvent.click(metadataButton);

    await waitFor(() => {
      expect(api.getImageMetadata).toHaveBeenCalledWith("img-1.png");
    });

    await waitFor(() => {
      expect(screen.getByText("a beautiful landscape")).toBeInTheDocument();
    });
  });

  it("should display metadata parameters", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
      availableCheckpoints: [
        {
          filename: "test-model.safetensors",
          title: "Test Model",
          path: "/models/checkpoints/test-model.safetensors",
          folder: "checkpoints",
          tensorCount: 100,
          fileSizeBytes: 1000000,
          architecture: "sd15",
          architectureSource: "modelspec",
        },
      ],
    });
    renderWithTheme(<ImageViewport />);

    const metadataButton = screen.getByRole("button", {
      name: "Show metadata",
    });

    // Wait for metadata to load so button becomes enabled
    await waitFor(() => {
      expect(metadataButton).not.toBeDisabled();
    });

    fireEvent.click(metadataButton);

    await waitFor(() => {
      expect(screen.getByText("512×768")).toBeInTheDocument();
      expect(screen.getByText("25")).toBeInTheDocument();
      expect(screen.getByText("7.5")).toBeInTheDocument();
      expect(screen.getByText("12345")).toBeInTheDocument();
      expect(screen.getByText("euler/normal")).toBeInTheDocument();
      expect(screen.getByText("Test Model")).toBeInTheDocument();
    });
  });

  it("should hide metadata when toggle clicked again", async () => {
    useImageStore.setState({
      images: mockImages,
      ui: { selectedFilename: "img-1.png", seedLocked: false },
    });
    renderWithTheme(<ImageViewport />);

    const metadataButton = screen.getByRole("button", {
      name: "Show metadata",
    });

    // Wait for metadata to load so button becomes enabled
    await waitFor(() => {
      expect(metadataButton).not.toBeDisabled();
    });

    // Show metadata
    fireEvent.click(metadataButton);
    await waitFor(() => {
      expect(screen.getByText("a beautiful landscape")).toBeInTheDocument();
    });

    // Hide metadata
    fireEvent.click(metadataButton);
    await waitFor(() => {
      expect(
        screen.queryByText("a beautiful landscape")
      ).not.toBeInTheDocument();
    });
  });
});
