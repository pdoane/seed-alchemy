import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Theme } from "@radix-ui/themes";
import { ParameterPanel } from "./ParameterPanel";
import { useImageStore } from "../store/imageStore";
import { defaultParams } from "../../shared/defaults";

// Mock the API client
vi.mock("../api/client", () => ({
  api: {
    getImages: vi.fn().mockResolvedValue([]),
    getModels: vi.fn().mockResolvedValue([]),
    getCheckpointNames: vi
      .fn()
      .mockResolvedValue(["model1.safetensors", "model2.safetensors"]),
    generate: vi.fn(),
    saveImage: vi.fn(),
    deleteImage: vi.fn(),
  },
}));

async function renderWithTheme(component: React.ReactNode) {
  const result = render(<Theme>{component}</Theme>);
  // Wait for async useEffect operations to complete
  await waitFor(() => {});
  return result;
}

describe("ParameterPanel", () => {
  beforeEach(() => {
    useImageStore.setState({
      params: { ...defaultParams },
      images: [],
      ui: { selectedFilename: null, seedLocked: false },
      historyIndex: -1,
      history: [],
      isGenerating: false,
      generationError: null,
      availableCheckpoints: [
        {
          filename: "model1.safetensors",
          path: "/models/checkpoints/model1.safetensors",
          folder: "checkpoints",
          tensorCount: 100,
          fileSizeBytes: 1000000,
          architecture: "sd15",
          architectureSource: "tensor",
          title: "model1",
        },
        {
          filename: "model2.safetensors",
          path: "/models/checkpoints/model2.safetensors",
          folder: "checkpoints",
          tensorCount: 100,
          fileSizeBytes: 1000000,
          architecture: "sdxl",
          architectureSource: "tensor",
          title: "model2",
        },
      ],
      availableLoras: [],
    });
  });

  it("should render the Image heading", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("Image")).toBeInTheDocument();
  });

  it("should render Generate button", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(
      screen.getByRole("button", { name: "Generate" })
    ).toBeInTheDocument();
  });

  it("should disable Generate button when no model selected", async () => {
    useImageStore.setState({
      params: { ...defaultParams, model: "" },
    });
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });

  it("should disable Generate button when prompt is empty", async () => {
    useImageStore.setState({
      params: { ...defaultParams, model: "test.safetensors", prompt: "" },
    });
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });

  it("should enable Generate button when model and prompt are set", async () => {
    useImageStore.setState({
      params: { ...defaultParams, model: "test.safetensors", prompt: "test" },
    });
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByRole("button", { name: "Generate" })).not.toBeDisabled();
  });

  it("should show Generating... when isGenerating is true", async () => {
    useImageStore.setState({ isGenerating: true });
    await renderWithTheme(<ParameterPanel />);
    expect(
      screen.getByRole("button", { name: "Generating..." })
    ).toBeInTheDocument();
  });

  it("should display generation error", async () => {
    useImageStore.setState({ generationError: "Test error message" });
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("Test error message")).toBeInTheDocument();
  });

  it("should update prompt when typing in textarea", async () => {
    await renderWithTheme(<ParameterPanel />);
    const promptInput = screen.getByPlaceholderText("Enter your prompt...");
    fireEvent.change(promptInput, { target: { value: "new prompt" } });

    expect(useImageStore.getState().params.prompt).toBe("new prompt");
  });

  it("should update negative prompt when typing", async () => {
    await renderWithTheme(<ParameterPanel />);
    const negativePromptInput = screen.getByPlaceholderText("What to avoid...");
    fireEvent.change(negativePromptInput, {
      target: { value: "ugly, blurry" },
    });

    expect(useImageStore.getState().params.negativePrompt).toBe("ugly, blurry");
  });

  it("should render size section", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("Size")).toBeInTheDocument();
    expect(screen.getByText("512×512")).toBeInTheDocument();
  });

  it("should render steps slider with current value", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("20")).toBeInTheDocument(); // default steps value
  });

  it("should render CFG Scale slider with current value", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("CFG")).toBeInTheDocument();
  });

  it("should render seed input with default value", async () => {
    await renderWithTheme(<ParameterPanel />);
    const seedInput = screen.getByRole("spinbutton");
    expect(seedInput).toHaveValue(1);
  });

  it("should render seed lock switch", async () => {
    await renderWithTheme(<ParameterPanel />);
    // There are multiple switches (Face Detailer, Upscale, Seed)
    const switches = screen.getAllByRole("switch");
    expect(switches.length).toBeGreaterThan(0);
  });

  it("should update seed when typing in input", async () => {
    await renderWithTheme(<ParameterPanel />);
    const seedInput = screen.getByRole("spinbutton");
    fireEvent.change(seedInput, { target: { value: "12345" } });
    expect(useImageStore.getState().params.seed).toBe(12345);
  });

  it("should render sampler settings section", async () => {
    await renderWithTheme(<ParameterPanel />);
    expect(screen.getByText("Sampler")).toBeInTheDocument();
  });
});
