import { create } from "zustand";
import type {
  EnhanceParams,
  EnhanceType,
  ImageParams,
} from "../../shared/types/image";
import type { ModelInfo, Architecture } from "../../shared/types/models";
import type {
  ControlNetConfig,
  ControlNetType,
} from "../../shared/types/controlnet";
import type {
  ProgressEvent,
  GenerationSession,
} from "../../shared/types/progress";
import type { ImageUiState } from "../../shared/types/state";
import { defaultParams } from "../../shared/defaults";
import { mergeWithDefaults } from "../../shared/merge";
import { filterCompatibleLoras, remapControlNets } from "../../shared/utils";
import { applyAspectRatio } from "../lib/dimensions";
import { api } from "../api/client";
import { toast } from "./toastStore";

export const defaultImageUi: ImageUiState = {
  selectedFilename: null,
  seedLocked: false,
};

// Context menu types
export type ThumbnailContext =
  | "gallery"
  | "source"
  | "reference"
  | "controlnet";

export interface ImageContextMenuState {
  filename: string;
  context: ThumbnailContext;
  position: { x: number; y: number };
  // Reference-specific props
  referenceIndex?: number;
  referenceCount?: number;
  onRemove?: () => void;
}

interface ImageState {
  // Current parameters for generation
  params: ImageParams;
  setParams: (params: Partial<ImageParams>) => void;
  resetParams: () => void;

  // Persisted UI state
  ui: ImageUiState;
  setUi: (ui: Partial<ImageUiState>) => void;

  // Image browser state (just filenames, sorted newest first)
  images: string[];
  setImages: (images: string[]) => void;
  addImage: (filename: string) => void;
  removeImage: (filename: string) => Promise<void>;
  loadImages: () => Promise<void>;

  // Navigation history
  historyIndex: number;
  history: string[];
  navigateTo: (filename: string | null) => void;
  navigateBack: () => void;
  navigateForward: () => void;

  // Generation state
  isGenerating: boolean;
  generationError: string | null;
  generate: (documentId?: string) => Promise<void>;
  enhance: (
    sourceImage: string,
    enhanceType: EnhanceType,
    documentId?: string
  ) => Promise<void>;
  cancelGeneration: () => Promise<void>;

  // Progress tracking (updated via SSE)
  activePromptId: string | null;
  generationStep: number;
  generationMaxSteps: number;
  generationNode: string | null;
  previewUrl: string | null;
  showPreview: boolean;
  setShowPreview: (show: boolean) => void;

  // Generation dimensions (from when generation started, for preview sizing)
  generationWidth: number;
  generationHeight: number;

  // Batch generation count (separate from params, not persisted)
  imageCount: number;
  setImageCount: (count: number) => void;

  // Browser grid columns (for keyboard navigation)
  browserColumns: number;
  setBrowserColumns: (columns: number) => void;

  // Available checkpoints (with metadata)
  availableCheckpoints: ModelInfo[];
  loadCheckpoints: () => Promise<void>;

  // Available LoRAs (with metadata)
  availableLoras: ModelInfo[];
  loadLoras: () => Promise<void>;

  // Get current model architecture
  getCurrentArchitecture: () => Architecture | null;

  // Set checkpoint and remove incompatible LoRAs
  setCheckpoint: (checkpoint: string) => void;

  // Reference image management
  addReferenceImage: (filename: string) => void;
  removeReferenceImage: (filename: string) => void;
  reorderReferenceImages: (fromIndex: number, toIndex: number) => void;

  // Source image management
  setSourceImage: (filename: string) => void;

  // ControlNet management
  addControlNet: (image: string, type?: ControlNetType) => void;
  updateControlNet: (index: number, updates: Partial<ControlNetConfig>) => void;
  removeControlNet: (index: number) => void;
  setControlNetImage: (index: number, image: string) => void;
  preprocessControlNet: (index: number) => Promise<void>;

  // Image context menu (shared across all thumbnails)
  contextMenu: ImageContextMenuState | null;
  openContextMenu: (state: ImageContextMenuState) => void;
  closeContextMenu: () => void;

  // Progress event handling (SSE events from server)
  handleProgressEvent: (event: ProgressEvent) => void;

  // State restoration (called by persistence layer)
  restoreGenerationSession: (session: GenerationSession) => void;
  restoreUserState: (
    params: Partial<ImageParams>,
    ui: Partial<ImageUiState>
  ) => void;
  resetForUserSwitch: () => void;

  // Prompt fragment insertion (for metadata display clickable fragments)
  insertPromptFragment: (text: string, target: "prompt" | "negative") => void;
}

export const useImageStore = create<ImageState>((set, get) => ({
  // Parameters
  params: { ...defaultParams },
  setParams: (newParams) =>
    set((state) => ({
      params: { ...state.params, ...newParams },
    })),
  resetParams: () => set({ params: { ...defaultParams } }),

  // Persisted UI state
  ui: { ...defaultImageUi },
  setUi: (ui: Partial<ImageUiState>) =>
    set((state) => ({ ui: { ...state.ui, ...ui } })),

  // Navigation with history tracking
  navigateTo: (filename: string | null) => {
    const state = get();
    if (filename && filename !== state.ui.selectedFilename) {
      // Push to history (truncate forward history)
      const newHistory = [
        ...state.history.slice(0, state.historyIndex + 1),
        filename,
      ];
      set({
        ui: { ...state.ui, selectedFilename: filename },
        history: newHistory,
        historyIndex: newHistory.length - 1,
      });
    } else {
      set({ ui: { ...state.ui, selectedFilename: filename } });
    }
  },

  // Images
  images: [],
  setImages: (images) => set({ images }),
  addImage: (filename) =>
    set((state) => ({
      images: [filename, ...state.images],
    })),
  removeImage: async (filename) => {
    try {
      await api.deleteImage(filename);
      set((state) => {
        const newImages = state.images.filter((img) => img !== filename);

        // Select next image if the deleted one was selected
        let newSelectedFilename = state.ui.selectedFilename;
        if (state.ui.selectedFilename === filename) {
          const deletedIndex = state.images.indexOf(filename);
          // Try to keep same index position, or fall back to last image
          newSelectedFilename =
            newImages[deletedIndex] ?? newImages[deletedIndex - 1] ?? null;
        }

        return {
          images: newImages,
          ui: { ...state.ui, selectedFilename: newSelectedFilename },
        };
      });
    } catch (error) {
      console.error("Failed to delete image:", error);
    }
  },
  loadImages: async () => {
    try {
      const images = await api.getImages();
      set({ images });
    } catch (error) {
      console.error("Failed to load images:", error);
    }
  },

  // Navigation history
  historyIndex: -1,
  history: [],
  navigateBack: () => {
    const { historyIndex, history, ui } = get();
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      set({
        historyIndex: newIndex,
        ui: { ...ui, selectedFilename: history[newIndex] ?? null },
      });
    }
  },
  navigateForward: () => {
    const { historyIndex, history, ui } = get();
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      set({
        historyIndex: newIndex,
        ui: { ...ui, selectedFilename: history[newIndex] ?? null },
      });
    }
  },

  // Generation
  isGenerating: false,
  generationError: null,

  generate: async (documentId?: string) => {
    const { imageCount, ui } = get();

    set({ generationError: null });

    // Generate random seed if not locked
    if (!ui.seedLocked) {
      const newSeed = Math.floor(Math.random() * 2 ** 32);
      set((state) => ({
        params: { ...state.params, seed: newSeed },
      }));
    }

    try {
      // Progress updates and completion handled via SSE in persistence.ts
      const { promptId } = await api.generate(
        get().params,
        imageCount,
        documentId
      );
      set({ activePromptId: promptId });
    } catch (error) {
      set({
        generationError:
          error instanceof Error ? error.message : "Generation failed",
      });
    }
  },

  enhance: async (
    sourceImage: string,
    enhanceType: EnhanceType,
    documentId?: string
  ) => {
    const { params, ui } = get();

    set({ generationError: null });

    // Generate random seed if not locked (for face detailer)
    const seed = ui.seedLocked
      ? params.seed
      : Math.floor(Math.random() * 2 ** 32);

    const enhanceParams: EnhanceParams = {
      source: sourceImage,
      model: params.model,
      sampler: params.sampler,
      scheduler: params.scheduler,
      steps: params.steps,
      cfgScale: params.cfgScale,
      seed,
      loras: params.loras,
      prompt: params.prompt,
      negativePrompt: params.negativePrompt,
      enhanceType,
      upscaleFactor: params.upscaleFactor,
    };

    try {
      const { promptId } = await api.enhance(enhanceParams, documentId);
      set({ activePromptId: promptId });
    } catch (error) {
      set({
        generationError:
          error instanceof Error ? error.message : "Enhance failed",
      });
    }
  },

  cancelGeneration: async () => {
    try {
      await api.cancelGeneration();
      // SSE will update isGenerating state when server confirms cancellation
    } catch {
      // Ignore errors - cancellation may fail if already complete
    }
  },

  // Progress tracking (updated via SSE subscription)
  activePromptId: null,
  generationStep: 0,
  generationMaxSteps: 0,
  generationNode: null,
  previewUrl: null,
  showPreview: true,
  setShowPreview: (show) => set({ showPreview: show }),

  // Generation dimensions (default to params, updated when generation starts)
  generationWidth: defaultParams.width,
  generationHeight: defaultParams.height,

  // Batch generation count
  imageCount: 1,
  setImageCount: (count) => set({ imageCount: count }),

  // Browser grid columns
  browserColumns: 2,
  setBrowserColumns: (columns) => set({ browserColumns: columns }),

  // Checkpoints (includes checkpoints, diffusion_models, and unet folders)
  availableCheckpoints: [],
  loadCheckpoints: async () => {
    try {
      const [checkpoints, diffusionModels, unetModels] = await Promise.all([
        api.getModels("checkpoints"),
        api.getModels("diffusion_models"),
        api.getModels("unet"),
      ]);
      set({
        availableCheckpoints: [
          ...checkpoints,
          ...diffusionModels,
          ...unetModels,
        ],
      });
    } catch (error) {
      console.error("Failed to load checkpoints:", error);
    }
  },

  // LoRAs
  availableLoras: [],
  loadLoras: async () => {
    try {
      const loras = await api.getModels("loras");
      set({ availableLoras: loras });
    } catch (error) {
      console.error("Failed to load LoRAs:", error);
    }
  },

  // Get current model architecture
  getCurrentArchitecture: () => {
    const { params, availableCheckpoints } = get();
    if (!params.model) return null;
    const checkpoint = availableCheckpoints.find(
      (m) => m.filename === params.model
    );
    return checkpoint?.architecture ?? null;
  },

  // Set checkpoint and adjust dimensions/LoRAs for architecture compatibility
  setCheckpoint: (checkpoint: string) => {
    const { availableCheckpoints, availableLoras, params } = get();

    // Find old and new architecture
    const oldInfo = availableCheckpoints.find(
      (c) => c.filename === params.model
    );
    const oldArch = oldInfo?.architecture ?? null;

    const newInfo = availableCheckpoints.find((c) => c.filename === checkpoint);
    const newArch = newInfo?.architecture ?? null;

    // Prepare updates
    const updates: Partial<ImageParams> = { model: checkpoint };

    // Adjust dimensions if architecture changed
    if (oldArch !== newArch && newArch && newArch !== "unknown") {
      const dims = applyAspectRatio(
        params.width,
        params.height,
        params.width,
        params.height,
        newArch
      );
      if (dims.width !== params.width || dims.height !== params.height) {
        updates.width = dims.width;
        updates.height = dims.height;
        toast.info(`Dimensions adjusted for ${newArch.toUpperCase()}`);
      }
    }

    // Filter existing LoRAs
    if (params.loras?.length) {
      const { compatible, incompatible } = filterCompatibleLoras(
        params.loras,
        newArch,
        availableLoras
      );
      if (incompatible.length > 0) {
        toast.warning(`Removed incompatible LoRAs: ${incompatible.join(", ")}`);
        updates.loras = compatible;
      }
    }

    // Remap existing ControlNets
    if (params.controlNets?.length && oldArch !== newArch) {
      const { remapped, removedCount } = remapControlNets(
        params.controlNets,
        newArch
      );
      if (removedCount > 0) {
        toast.warning(`Removed ${removedCount} incompatible ControlNet(s)`);
      }
      if (
        removedCount > 0 ||
        remapped.some((cn, i) => cn.type !== params.controlNets![i]?.type)
      ) {
        updates.controlNets = remapped;
      }
    }

    set((state) => ({
      params: { ...state.params, ...updates },
    }));
  },

  // Reference image management
  addReferenceImage: (filename: string) => {
    set((state) => {
      const current = state.params.referenceImages || [];
      // Don't add if already present
      if (current.some((r) => r.filename === filename)) {
        return state;
      }
      return {
        params: {
          ...state.params,
          referenceImages: [...current, { filename }],
        },
      };
    });
  },

  removeReferenceImage: (filename: string) => {
    set((state) => ({
      params: {
        ...state.params,
        referenceImages: (state.params.referenceImages || []).filter(
          (r) => r.filename !== filename
        ),
      },
    }));
  },

  reorderReferenceImages: (fromIndex: number, toIndex: number) => {
    set((state) => {
      const images = [...(state.params.referenceImages || [])];
      const [moved] = images.splice(fromIndex, 1);
      if (moved) {
        images.splice(toIndex, 0, moved);
      }
      return {
        params: {
          ...state.params,
          referenceImages: images,
        },
      };
    });
  },

  // Source image management
  setSourceImage: (filename: string) => {
    set((state) => ({
      params: {
        ...state.params,
        sourceImage: filename,
      },
    }));
  },

  // ControlNet management
  addControlNet: (image: string, type: ControlNetType = "lineart") => {
    set((state) => {
      const current = state.params.controlNets || [];
      // Default preprocessor based on type
      let defaultPreprocessor: string | null = null;
      if (type === "lineart" || type === "canny") {
        defaultPreprocessor = "CannyEdgePreprocessor";
      }
      const newConfig: ControlNetConfig = {
        type,
        image,
        preprocessor: defaultPreprocessor,
        weight: 1,
      };
      return {
        params: {
          ...state.params,
          controlNets: [...current, newConfig],
        },
      };
    });
  },

  updateControlNet: (index: number, updates: Partial<ControlNetConfig>) => {
    set((state) => {
      const controlNets = [...(state.params.controlNets || [])];
      if (index >= 0 && index < controlNets.length && controlNets[index]) {
        controlNets[index] = {
          ...controlNets[index],
          ...updates,
        } as ControlNetConfig;
      }
      return {
        params: {
          ...state.params,
          controlNets,
        },
      };
    });
  },

  removeControlNet: (index: number) => {
    set((state) => {
      const controlNets = [...(state.params.controlNets || [])];
      controlNets.splice(index, 1);
      return {
        params: {
          ...state.params,
          controlNets,
        },
      };
    });
  },

  setControlNetImage: (index: number, image: string) => {
    const { params } = get();
    const controlNets = params.controlNets || [];

    if (index >= 0 && index < controlNets.length) {
      // Update existing ControlNet
      get().updateControlNet(index, { image });
    } else {
      // Create new ControlNet
      get().addControlNet(image);
    }
  },

  preprocessControlNet: async (index: number) => {
    const { params } = get();
    const controlNets = params.controlNets || [];
    const config = controlNets[index];

    if (!config || !config.preprocessor) {
      toast.error("No preprocessor selected");
      return;
    }

    try {
      const resolution = Math.min(params.width, params.height);
      await api.preprocess(config.image, config.preprocessor, resolution);
      toast.info("Preprocessing started");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Preprocessing failed"
      );
    }
  },

  // Context menu
  contextMenu: null,
  openContextMenu: (menuState) => set({ contextMenu: menuState }),
  closeContextMenu: () => set({ contextMenu: null }),

  // Progress event handling
  handleProgressEvent: (event: ProgressEvent) => {
    const store = get();
    const { activePromptId } = store;

    // Ignore events from stale generations
    if (activePromptId && event.promptId !== activePromptId) {
      return;
    }

    switch (event.type) {
      case "execution_start": {
        const { params } = store;
        set({
          isGenerating: true,
          generationStep: 0,
          generationMaxSteps: 0,
          generationNode: null,
          previewUrl: null,
          generationWidth: params.width,
          generationHeight: params.height,
        });
        break;
      }

      case "executing":
        set({ generationNode: event.node });
        break;

      case "progress":
        set({
          isGenerating: true,
          generationStep: event.step,
          generationMaxSteps: event.maxSteps,
        });
        break;

      case "preview":
        set({
          isGenerating: true,
          previewUrl: event.previewUrl,
        });
        break;

      case "execution_success":
        // Skip document mode - handled by documentStore
        if (event.documentId) {
          set({
            activePromptId: null,
            isGenerating: false,
            generationStep: 0,
            generationMaxSteps: 0,
            generationNode: null,
            previewUrl: null,
          });
          break;
        }
        // Image mode: add completed images to browser
        for (const img of event.images) {
          store.addImage(img.filename);
        }
        // Select the first new image
        if (event.images[0]) {
          store.navigateTo(event.images[0].filename);
        }
        set({
          activePromptId: null,
          isGenerating: false,
          generationStep: 0,
          generationMaxSteps: 0,
          generationNode: null,
          previewUrl: null,
        });
        break;

      case "execution_interrupted":
        set({
          activePromptId: null,
          isGenerating: false,
          generationStep: 0,
          generationMaxSteps: 0,
          generationNode: null,
          previewUrl: null,
        });
        break;

      case "execution_error":
        set({
          activePromptId: null,
          isGenerating: false,
          generationStep: 0,
          generationMaxSteps: 0,
          generationNode: null,
          previewUrl: null,
          generationError: event.error,
        });
        break;
    }
  },
  // State restoration (called by persistence layer)
  restoreGenerationSession: (session) =>
    set({
      isGenerating: true,
      generationStep: session.step,
      generationMaxSteps: session.maxSteps,
      generationNode: session.node,
      previewUrl: session.previewUrl,
      generationWidth: session.width,
      generationHeight: session.height,
    }),

  restoreUserState: (params, ui) =>
    set({
      params: mergeWithDefaults(defaultParams, params),
      ui: mergeWithDefaults(defaultImageUi, ui),
    }),

  resetForUserSwitch: () =>
    set({
      images: [],
      params: { ...defaultParams },
      ui: { ...defaultImageUi },
      historyIndex: -1,
      history: [],
      isGenerating: false,
      generationError: null,
      generationStep: 0,
      generationMaxSteps: 0,
      generationNode: null,
      previewUrl: null,
    }),

  // Prompt fragment insertion - appends fragment to prompt/negative prompt
  insertPromptFragment: (text, target) =>
    set((state) => {
      const key = target === "prompt" ? "prompt" : "negativePrompt";
      const current = state.params[key];
      const newValue = current ? `${current}\n${text}` : text;
      return {
        params: {
          ...state.params,
          [key]: newValue,
        },
      };
    }),
}));

// Selectors
export const selectSelectedImage = (state: ImageState) =>
  state.images.find((img) => img === state.ui.selectedFilename);

export const selectCanNavigateBack = (state: ImageState) =>
  state.historyIndex > 0;

export const selectCanNavigateForward = (state: ImageState) =>
  state.historyIndex < state.history.length - 1;
