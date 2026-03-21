import { create } from "zustand";
import type {
  Asset,
  Document,
  DocumentSummary,
  Layer,
  LayerGroup,
  CanvasSize,
} from "../../shared/types/document";
import type { BlendMode } from "../../shared/types/image";
import type { ProgressEvent } from "../../shared/types/progress";
import type { DocumentUiState } from "../../shared/types/state";
import { api } from "../api/client";

export const defaultDocumentUi: DocumentUiState = {
  currentDocumentId: null,
  activeTool: "select",
};

interface DocumentState {
  // List of documents (summaries for efficiency)
  documents: DocumentSummary[];
  loadDocuments: () => Promise<void>;

  // Currently open document (full details)
  currentDocument: Document | null;
  loadDocument: (id: string) => Promise<void>;
  closeDocument: () => void;

  // Document CRUD
  createDocument: (name?: string, canvasSize?: CanvasSize) => Promise<Document>;
  updateDocument: (updates: Partial<Document>) => Promise<void>;
  deleteDocument: (id: string) => Promise<void>;

  // Persisted UI state
  documentUi: DocumentUiState;
  setDocumentUi: (ui: Partial<DocumentUiState>) => void;

  // Selected layer (stored in currentDocument.selectedLayerId)
  setSelectedLayerId: (id: string | null) => void;

  // Pending target layer for generation (captured when generate starts)
  pendingTargetLayerId: string | null;
  setPendingTargetLayerId: (id: string | null) => void;

  // Layer operations (typeless layers)
  addLayer: (name: string, assetId?: string) => void;
  updateLayer: (layerId: string, updates: Partial<Layer>) => void;
  removeLayer: (layerId: string) => void;
  toggleLayerVisibility: (layerId: string) => void;
  reorderLayers: (fromIndex: number, toIndex: number) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  setLayerBlendMode: (layerId: string, blendMode: BlendMode) => void;
  setLayerPosition: (layerId: string, x: number, y: number) => void;
  setLayerAsset: (layerId: string, assetId: string | undefined) => void;

  // Asset operations
  refreshAssets: () => Promise<void>;
  getAssetByFilename: (filename: string) => Asset | undefined;

  // Progress event handling (SSE events from server)
  handleProgressEvent: (event: ProgressEvent) => Promise<void>;

  // State restoration (called by persistence layer)
  restoreUserState: (documentUi: Partial<DocumentUiState>) => Promise<void>;
  resetForUserSwitch: () => void;
}

// Generate a unique ID
function generateId(): string {
  return crypto.randomUUID();
}

// Helper to find layer by ID (handles nested layer groups)
function findLayerById(
  items: (Layer | LayerGroup)[],
  id: string
): Layer | null {
  for (const item of items) {
    if (item.id === id && !("children" in item)) {
      return item as Layer;
    }
    if ("children" in item) {
      const found = findLayerById(item.children, id);
      if (found) return found;
    }
  }
  return null;
}

// Helper to update layer by ID (handles nested layer groups)
function updateLayerById(
  items: (Layer | LayerGroup)[],
  id: string,
  updates: Partial<Layer>
): (Layer | LayerGroup)[] {
  return items.map((item) => {
    if (item.id === id && !("children" in item)) {
      return { ...item, ...updates };
    }
    if ("children" in item) {
      return { ...item, children: updateLayerById(item.children, id, updates) };
    }
    return item;
  });
}

// Helper to filter out layer by ID (handles nested layer groups)
function removeLayerById(
  items: (Layer | LayerGroup)[],
  id: string
): (Layer | LayerGroup)[] {
  return items
    .filter((item) => item.id !== id)
    .map((item) => {
      if ("children" in item) {
        return { ...item, children: removeLayerById(item.children, id) };
      }
      return item;
    });
}

export const useDocumentStore = create<DocumentState>((set, get) => ({
  // Documents list
  documents: [],
  loadDocuments: async () => {
    try {
      const documents = await api.getDocuments();
      set({ documents });
    } catch (error) {
      console.error("Failed to load documents:", error);
    }
  },

  // Persisted UI state
  documentUi: { ...defaultDocumentUi },
  setDocumentUi: (ui: Partial<DocumentUiState>) =>
    set((state) => ({ documentUi: { ...state.documentUi, ...ui } })),

  // Current document
  currentDocument: null,
  loadDocument: async (id: string) => {
    try {
      const doc = await api.getDocument(id);
      // Use saved selectedLayerId, or fall back to first layer if invalid/null
      let selectedLayerId = doc.selectedLayerId;
      if (!selectedLayerId || !findLayerById(doc.layers, selectedLayerId)) {
        const firstLayer = doc.layers.find((l) => !("children" in l));
        selectedLayerId = firstLayer?.id ?? null;
        // Update document with valid selection
        if (selectedLayerId !== doc.selectedLayerId) {
          doc.selectedLayerId = selectedLayerId;
          await api.updateDocument(doc.id, { selectedLayerId });
        }
      }
      set((state) => ({
        currentDocument: doc,
        documentUi: { ...state.documentUi, currentDocumentId: doc.id },
      }));
    } catch (error) {
      console.error("Failed to load document:", error);
    }
  },
  closeDocument: () =>
    set((state) => ({
      currentDocument: null,
      documentUi: { ...state.documentUi, currentDocumentId: null },
    })),

  // Document CRUD
  createDocument: async (name?: string, canvasSize?: CanvasSize) => {
    const doc = await api.createDocument(name, canvasSize);
    // Refresh document list
    await get().loadDocuments();
    // Open the new document
    set((state) => ({
      currentDocument: doc,
      documentUi: { ...state.documentUi, currentDocumentId: doc.id },
    }));
    return doc;
  },

  updateDocument: async (updates: Partial<Document>) => {
    const { currentDocument } = get();
    if (!currentDocument) return;

    try {
      // Persist to server but don't overwrite local state - local state is source of truth
      // This prevents race conditions with rapid updates (e.g., multiple addLayer calls)
      await api.updateDocument(currentDocument.id, updates);
      // Refresh the summary list for sidebar
      await get().loadDocuments();
    } catch (error) {
      console.error("Failed to update document:", error);
    }
  },

  deleteDocument: async (id: string) => {
    try {
      await api.deleteDocument(id);
      const { currentDocument } = get();
      if (currentDocument?.id === id) {
        set((state) => ({
          currentDocument: null,
          documentUi: { ...state.documentUi, currentDocumentId: null },
        }));
      }
      await get().loadDocuments();
    } catch (error) {
      console.error("Failed to delete document:", error);
    }
  },

  // Selected layer
  setSelectedLayerId: (id) => {
    const { currentDocument, updateDocument } = get();
    if (!currentDocument) return;

    set({
      currentDocument: { ...currentDocument, selectedLayerId: id },
    });

    // Persist to server
    updateDocument({ selectedLayerId: id });
  },

  // Pending target layer for generation
  pendingTargetLayerId: null,
  setPendingTargetLayerId: (id) => set({ pendingTargetLayerId: id }),

  // Layer operations (typeless layers)
  addLayer: (name: string, assetId?: string) => {
    const { currentDocument, updateDocument } = get();
    if (!currentDocument) return;

    const newLayer: Layer = {
      id: generateId(),
      name,
      asset: assetId,
      visible: true,
      opacity: 100,
      blendMode: "normal",
      position: { x: 0, y: 0 },
    };

    const layers = [...currentDocument.layers, newLayer];
    set({
      currentDocument: {
        ...currentDocument,
        layers,
        selectedLayerId: newLayer.id,
      },
    });

    // Persist
    updateDocument({ layers, selectedLayerId: newLayer.id });
  },

  updateLayer: (layerId: string, updates: Partial<Layer>) => {
    const { currentDocument, updateDocument } = get();
    if (!currentDocument) return;

    const layers = updateLayerById(currentDocument.layers, layerId, updates);
    set({ currentDocument: { ...currentDocument, layers } });

    // Persist
    updateDocument({ layers });
  },

  removeLayer: (layerId: string) => {
    const { currentDocument, updateDocument } = get();
    if (!currentDocument) return;

    const layers = removeLayerById(currentDocument.layers, layerId);
    const firstLayer = layers.find((l) => !("children" in l));
    const newSelectedId =
      currentDocument.selectedLayerId === layerId
        ? (firstLayer?.id ?? null)
        : currentDocument.selectedLayerId;

    set({
      currentDocument: {
        ...currentDocument,
        layers,
        selectedLayerId: newSelectedId,
      },
    });

    // Persist
    updateDocument({ layers, selectedLayerId: newSelectedId });
  },

  toggleLayerVisibility: (layerId: string) => {
    const { currentDocument } = get();
    if (!currentDocument) return;

    const layer = findLayerById(currentDocument.layers, layerId);
    if (layer) {
      get().updateLayer(layerId, { visible: !layer.visible });
    }
  },

  reorderLayers: (fromIndex: number, toIndex: number) => {
    const { currentDocument, updateDocument } = get();
    if (!currentDocument) return;

    const layers = [...currentDocument.layers];
    const [moved] = layers.splice(fromIndex, 1);
    if (moved) {
      layers.splice(toIndex, 0, moved);
      set({ currentDocument: { ...currentDocument, layers } });

      // Persist
      updateDocument({ layers });
    }
  },

  setLayerOpacity: (layerId: string, opacity: number) => {
    get().updateLayer(layerId, {
      opacity: Math.max(0, Math.min(100, opacity)),
    });
  },

  setLayerBlendMode: (layerId: string, blendMode: BlendMode) => {
    get().updateLayer(layerId, { blendMode });
  },

  setLayerPosition: (layerId: string, x: number, y: number) => {
    get().updateLayer(layerId, { position: { x, y } });
  },

  setLayerAsset: (layerId: string, assetId: string | undefined) => {
    get().updateLayer(layerId, { asset: assetId });
  },

  // Asset operations
  refreshAssets: async () => {
    const { currentDocument } = get();
    if (!currentDocument) return;

    // Reload document from server to get updated assets from PNG files
    const doc = await api.getDocument(currentDocument.id);
    set({ currentDocument: doc });
  },

  getAssetByFilename: (filename: string) => {
    const { currentDocument } = get();
    if (!currentDocument) return undefined;
    return currentDocument.assets.find((a) => a.filename === filename);
  },

  // Progress event handling
  handleProgressEvent: async (event: ProgressEvent) => {
    // Only handle execution_success with documentId
    if (event.type !== "execution_success" || !event.documentId) {
      return;
    }

    const { currentDocument, pendingTargetLayerId, loadDocument } = get();

    // Only handle if this is for the currently open document
    if (currentDocument?.id !== event.documentId) {
      return;
    }

    const images = event.images;
    if (images.length === 0) {
      return;
    }

    const targetLayerId = pendingTargetLayerId;

    // Reload document to get new assets
    await loadDocument(event.documentId);

    const refreshedDoc = get().currentDocument;
    if (!refreshedDoc) {
      return;
    }

    if (targetLayerId) {
      // First image updates the target layer
      const firstImage = images[0];
      if (firstImage?.assetId) {
        get().setLayerAsset(targetLayerId, firstImage.assetId);
      }
      // Remaining images create new layers
      for (let i = 1; i < images.length; i++) {
        const img = images[i];
        if (img?.assetId) {
          const layerCount = get().currentDocument?.layers.length ?? 0;
          get().addLayer(`Layer ${layerCount + 1}`, img.assetId);
        }
      }
    } else {
      // No target layer - create new layers for all images
      for (const img of images) {
        if (img?.assetId) {
          const layerCount = get().currentDocument?.layers.length ?? 0;
          get().addLayer(`Layer ${layerCount + 1}`, img.assetId);
        }
      }
    }

    // Clear the pending target
    get().setPendingTargetLayerId(null);
  },

  // State restoration (called by persistence layer)
  restoreUserState: async (documentUi: Partial<DocumentUiState>) => {
    // Restore UI state
    set({
      documentUi: { ...defaultDocumentUi, ...documentUi },
    });

    // Load the document if one was open
    if (documentUi.currentDocumentId) {
      try {
        await get().loadDocument(documentUi.currentDocumentId);
      } catch (error) {
        console.error("Failed to restore document:", error);
      }
    }
  },

  resetForUserSwitch: () =>
    set({
      documents: [],
      currentDocument: null,
      documentUi: { ...defaultDocumentUi },
      pendingTargetLayerId: null,
    }),
}));

// Selectors
export const selectCurrentDocument = (state: DocumentState) =>
  state.currentDocument;

export const selectSelectedLayer = (state: DocumentState) => {
  const { currentDocument } = state;
  if (!currentDocument || !currentDocument.selectedLayerId) return null;
  return findLayerById(currentDocument.layers, currentDocument.selectedLayerId);
};

export const selectLayers = (state: DocumentState) =>
  state.currentDocument?.layers ?? [];

export const selectAssets = (state: DocumentState) =>
  state.currentDocument?.assets ?? [];
