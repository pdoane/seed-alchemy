import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  useDocumentStore,
  selectCurrentDocument,
  selectSelectedLayer,
  selectLayers,
} from "./documentStore";
import type { Document, DocumentSummary } from "../../shared/types/document";
import type { ProgressEvent } from "../../shared/types/progress";

// Mock the API client
vi.mock("../api/client", () => ({
  api: {
    getDocuments: vi.fn().mockResolvedValue([]),
    getDocument: vi.fn(),
    createDocument: vi.fn(),
    updateDocument: vi.fn().mockResolvedValue({}),
    deleteDocument: vi.fn(),
  },
}));

import { api } from "../api/client";

const createMockDocument = (overrides: Partial<Document> = {}): Document => ({
  id: "doc-1",
  name: "Test Document",
  canvasSize: { width: 1024, height: 1024 },
  layers: [],
  assets: [],
  createdAt: "2024-01-01T00:00:00Z",
  modifiedAt: "2024-01-01T00:00:00Z",
  selectedLayerId: null,
  ...overrides,
});

const createMockSummary = (
  overrides: Partial<DocumentSummary> = {}
): DocumentSummary => ({
  id: "doc-1",
  name: "Test Document",
  canvasSize: { width: 1024, height: 1024 },
  layerCount: 0,
  modifiedAt: "2024-01-01T00:00:00Z",
  ...overrides,
});

describe("documentStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store to initial state
    useDocumentStore.setState({
      documents: [],
      currentDocument: null,
      pendingTargetLayerId: null,
      documentUi: {
        currentDocumentId: null,
        activeTool: "select",
      },
    });
  });

  describe("loadDocuments", () => {
    it("should load documents from API", async () => {
      const mockDocs = [createMockSummary({ id: "doc-1" })];
      vi.mocked(api.getDocuments).mockResolvedValue(mockDocs);

      await useDocumentStore.getState().loadDocuments();

      expect(api.getDocuments).toHaveBeenCalled();
      expect(useDocumentStore.getState().documents).toEqual(mockDocs);
    });

    it("should handle API errors gracefully", async () => {
      vi.mocked(api.getDocuments).mockRejectedValue(new Error("Network error"));
      const consoleSpy = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await useDocumentStore.getState().loadDocuments();

      expect(consoleSpy).toHaveBeenCalled();
      expect(useDocumentStore.getState().documents).toEqual([]);

      consoleSpy.mockRestore();
    });
  });

  describe("loadDocument", () => {
    it("should load document and select first layer", async () => {
      const mockDoc = createMockDocument({
        layers: [
          {
            id: "layer-1",
            name: "Layer 1",
            visible: true,
            opacity: 100,
            blendMode: "normal",
            position: { x: 0, y: 0 },
          },
        ],
      });
      vi.mocked(api.getDocument).mockResolvedValue(mockDoc);

      await useDocumentStore.getState().loadDocument("doc-1");

      expect(api.getDocument).toHaveBeenCalledWith("doc-1");
      expect(useDocumentStore.getState().currentDocument?.selectedLayerId).toBe(
        "layer-1"
      );
    });

    it("should set selectedLayerId to null if no layers", async () => {
      const mockDoc = createMockDocument({ layers: [] });
      vi.mocked(api.getDocument).mockResolvedValue(mockDoc);

      await useDocumentStore.getState().loadDocument("doc-1");

      expect(
        useDocumentStore.getState().currentDocument?.selectedLayerId
      ).toBeNull();
    });
  });

  describe("closeDocument", () => {
    it("should clear current document and selection", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument({ selectedLayerId: "layer-1" }),
      });

      useDocumentStore.getState().closeDocument();

      expect(useDocumentStore.getState().currentDocument).toBeNull();
    });
  });

  describe("layer operations", () => {
    beforeEach(() => {
      vi.mocked(api.getDocuments).mockResolvedValue([]);
    });

    it("addLayer should add new layer and select it", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument(),
      });

      useDocumentStore.getState().addLayer("New Layer");

      const state = useDocumentStore.getState();
      expect(state.currentDocument?.layers).toHaveLength(1);
      expect(state.currentDocument?.layers[0]?.name).toBe("New Layer");
      expect(state.currentDocument?.selectedLayerId).toBe(
        state.currentDocument?.layers[0]?.id
      );
    });

    it("addLayer should include assetId if provided", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument(),
      });

      useDocumentStore.getState().addLayer("New Layer", "asset-123");

      const layer = useDocumentStore.getState().currentDocument?.layers[0];
      // Type guard to check it's a Layer not a LayerGroup
      if (layer && !("children" in layer)) {
        expect(layer.asset).toBe("asset-123");
      } else {
        throw new Error("Expected layer to be a Layer, not LayerGroup");
      }
    });

    it("removeLayer should remove layer and update selection", () => {
      const layer1 = {
        id: "layer-1",
        name: "Layer 1",
        visible: true,
        opacity: 100,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      const layer2 = {
        id: "layer-2",
        name: "Layer 2",
        visible: true,
        opacity: 100,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      useDocumentStore.setState({
        currentDocument: createMockDocument({
          layers: [layer1, layer2],
          selectedLayerId: "layer-1",
        }),
      });

      useDocumentStore.getState().removeLayer("layer-1");

      const state = useDocumentStore.getState();
      expect(state.currentDocument?.layers).toHaveLength(1);
      expect(state.currentDocument?.layers[0]?.id).toBe("layer-2");
      expect(state.currentDocument?.selectedLayerId).toBe("layer-2");
    });

    it("toggleLayerVisibility should toggle visible flag", () => {
      const layer = {
        id: "layer-1",
        name: "Layer 1",
        visible: true,
        opacity: 100,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      useDocumentStore.setState({
        currentDocument: createMockDocument({ layers: [layer] }),
      });

      useDocumentStore.getState().toggleLayerVisibility("layer-1");

      expect(
        useDocumentStore.getState().currentDocument?.layers[0]?.visible
      ).toBe(false);
    });

    it("setLayerOpacity should clamp value between 0 and 100", () => {
      const layer = {
        id: "layer-1",
        name: "Layer 1",
        visible: true,
        opacity: 50,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      useDocumentStore.setState({
        currentDocument: createMockDocument({ layers: [layer] }),
      });

      useDocumentStore.getState().setLayerOpacity("layer-1", 150);
      expect(
        useDocumentStore.getState().currentDocument?.layers[0]?.opacity
      ).toBe(100);

      useDocumentStore.getState().setLayerOpacity("layer-1", -10);
      expect(
        useDocumentStore.getState().currentDocument?.layers[0]?.opacity
      ).toBe(0);
    });

    it("reorderLayers should move layer to new position", () => {
      const layers = [
        {
          id: "layer-1",
          name: "Layer 1",
          visible: true,
          opacity: 100,
          blendMode: "normal" as const,
          position: { x: 0, y: 0 },
        },
        {
          id: "layer-2",
          name: "Layer 2",
          visible: true,
          opacity: 100,
          blendMode: "normal" as const,
          position: { x: 0, y: 0 },
        },
        {
          id: "layer-3",
          name: "Layer 3",
          visible: true,
          opacity: 100,
          blendMode: "normal" as const,
          position: { x: 0, y: 0 },
        },
      ];
      useDocumentStore.setState({
        currentDocument: createMockDocument({ layers }),
      });

      useDocumentStore.getState().reorderLayers(0, 2);

      const reorderedLayers =
        useDocumentStore.getState().currentDocument?.layers;
      expect(reorderedLayers?.map((l) => l.id)).toEqual([
        "layer-2",
        "layer-3",
        "layer-1",
      ]);
    });
  });

  describe("selectedLayerId", () => {
    it("should set selected layer ID", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument(),
      });
      useDocumentStore.getState().setSelectedLayerId("layer-1");
      expect(useDocumentStore.getState().currentDocument?.selectedLayerId).toBe(
        "layer-1"
      );
    });

    it("should clear selected layer ID", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument({ selectedLayerId: "layer-1" }),
      });
      useDocumentStore.getState().setSelectedLayerId(null);
      expect(
        useDocumentStore.getState().currentDocument?.selectedLayerId
      ).toBeNull();
    });
  });

  describe("pendingTargetLayerId", () => {
    it("should set pending target layer ID", () => {
      useDocumentStore.getState().setPendingTargetLayerId("layer-1");
      expect(useDocumentStore.getState().pendingTargetLayerId).toBe("layer-1");
    });

    it("should clear pending target layer ID", () => {
      useDocumentStore.setState({ pendingTargetLayerId: "layer-1" });
      useDocumentStore.getState().setPendingTargetLayerId(null);
      expect(useDocumentStore.getState().pendingTargetLayerId).toBeNull();
    });
  });

  describe("activeTool", () => {
    it("should set active tool", () => {
      useDocumentStore.getState().setDocumentUi({ activeTool: "brush" });
      expect(useDocumentStore.getState().documentUi.activeTool).toBe("brush");
    });
  });

  describe("selectSelectedLayer", () => {
    it("should return selected layer", () => {
      const layer = {
        id: "layer-1",
        name: "Layer 1",
        visible: true,
        opacity: 100,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      useDocumentStore.setState({
        currentDocument: createMockDocument({
          layers: [layer],
          selectedLayerId: "layer-1",
        }),
      });

      expect(selectSelectedLayer(useDocumentStore.getState())).toEqual(layer);
    });

    it("should return null if no document", () => {
      useDocumentStore.setState({
        currentDocument: null,
      });

      expect(selectSelectedLayer(useDocumentStore.getState())).toBeNull();
    });

    it("should return null if no selection", () => {
      useDocumentStore.setState({
        currentDocument: createMockDocument({ selectedLayerId: null }),
      });

      expect(selectSelectedLayer(useDocumentStore.getState())).toBeNull();
    });
  });

  describe("selectors", () => {
    it("selectCurrentDocument should return current document", () => {
      const doc = createMockDocument();
      useDocumentStore.setState({ currentDocument: doc });

      expect(selectCurrentDocument(useDocumentStore.getState())).toEqual(doc);
    });

    it("selectLayers should return layers or empty array", () => {
      expect(selectLayers(useDocumentStore.getState())).toEqual([]);

      const layers = [
        {
          id: "layer-1",
          name: "Layer 1",
          visible: true,
          opacity: 100,
          blendMode: "normal" as const,
          position: { x: 0, y: 0 },
        },
      ];
      useDocumentStore.setState({
        currentDocument: createMockDocument({ layers }),
      });

      expect(selectLayers(useDocumentStore.getState())).toEqual(layers);
    });
  });

  describe("handleProgressEvent", () => {
    beforeEach(() => {
      vi.mocked(api.getDocuments).mockResolvedValue([]);
    });

    it("should ignore non-execution_success events", async () => {
      const mockDoc = createMockDocument({ id: "doc-1", layers: [] });
      useDocumentStore.setState({
        currentDocument: mockDoc,
        pendingTargetLayerId: "layer-1",
      });

      const progressEvent: ProgressEvent = {
        type: "progress",
        promptId: "prompt-1",
        step: 10,
        maxSteps: 20,
      };

      await useDocumentStore.getState().handleProgressEvent(progressEvent);

      // State should be unchanged
      expect(useDocumentStore.getState().pendingTargetLayerId).toBe("layer-1");
    });

    it("should ignore execution_success without documentId", async () => {
      const mockDoc = createMockDocument({ id: "doc-1", layers: [] });
      useDocumentStore.setState({
        currentDocument: mockDoc,
        pendingTargetLayerId: "layer-1",
      });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [{ filename: "img-1.png" }],
      };

      await useDocumentStore.getState().handleProgressEvent(event);

      // State should be unchanged - no documentId means image mode
      expect(useDocumentStore.getState().pendingTargetLayerId).toBe("layer-1");
    });

    it("should ignore execution_success for different document", async () => {
      const mockDoc = createMockDocument({ id: "doc-1", layers: [] });
      useDocumentStore.setState({
        currentDocument: mockDoc,
        pendingTargetLayerId: "layer-1",
      });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [{ filename: "img-1.png", assetId: "asset-1" }],
        documentId: "doc-2", // Different document
      };

      await useDocumentStore.getState().handleProgressEvent(event);

      // State should be unchanged
      expect(useDocumentStore.getState().pendingTargetLayerId).toBe("layer-1");
    });

    it("should update target layer with first image asset", async () => {
      const existingLayer = {
        id: "layer-1",
        name: "Layer 1",
        visible: true,
        opacity: 100,
        blendMode: "normal" as const,
        position: { x: 0, y: 0 },
      };
      const mockDoc = createMockDocument({
        id: "doc-1",
        layers: [existingLayer],
        assets: [],
      });

      // Mock getDocument to return document with new asset
      const updatedDoc = createMockDocument({
        id: "doc-1",
        layers: [existingLayer],
        assets: [{ filename: "img-1.png", metadata: {} as never }],
      });
      vi.mocked(api.getDocument).mockResolvedValue(updatedDoc);

      useDocumentStore.setState({
        currentDocument: { ...mockDoc, selectedLayerId: "layer-1" },
        pendingTargetLayerId: "layer-1",
      });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [{ filename: "img-1.png", assetId: "asset-1" }],
        documentId: "doc-1",
      };

      await useDocumentStore.getState().handleProgressEvent(event);

      const state = useDocumentStore.getState();
      // Target layer should have new asset
      const layer = state.currentDocument?.layers[0];
      if (layer && !("children" in layer)) {
        expect(layer.asset).toBe("asset-1");
      }
      // Pending target should be cleared
      expect(state.pendingTargetLayerId).toBeNull();
    });

    it("should create new layers when no target layer", async () => {
      const mockDoc = createMockDocument({
        id: "doc-1",
        layers: [],
        assets: [],
      });

      // Mock getDocument to return document with new assets
      const updatedDoc = createMockDocument({
        id: "doc-1",
        layers: [],
        assets: [
          { filename: "img-1.png", metadata: {} as never },
          { filename: "img-2.png", metadata: {} as never },
        ],
      });
      vi.mocked(api.getDocument).mockResolvedValue(updatedDoc);

      useDocumentStore.setState({
        currentDocument: mockDoc,
        pendingTargetLayerId: null,
      });

      const event: ProgressEvent = {
        type: "execution_success",
        promptId: "prompt-1",
        images: [
          { filename: "img-1.png", assetId: "asset-1" },
          { filename: "img-2.png", assetId: "asset-2" },
        ],
        documentId: "doc-1",
      };

      await useDocumentStore.getState().handleProgressEvent(event);

      // Verify loadDocument was called
      expect(api.getDocument).toHaveBeenCalledWith("doc-1");

      const state = useDocumentStore.getState();
      // Should have created 2 new layers (addLayer is called for each image with assetId)
      expect(state.currentDocument?.layers.length).toBe(2);
      expect(state.pendingTargetLayerId).toBeNull();
    });
  });

  describe("resetForUserSwitch", () => {
    it("should reset state for user switch", () => {
      const mockDoc = createMockDocument({
        id: "doc-1",
        layers: [
          {
            id: "layer-1",
            name: "Layer 1",
            visible: true,
            opacity: 100,
            blendMode: "normal",
            position: { x: 0, y: 0 },
          },
        ],
      });

      useDocumentStore.setState({
        documents: [
          {
            id: "doc-1",
            name: "Test Doc",
            canvasSize: { width: 1024, height: 1024 },
            modifiedAt: "2024-01-01",
            layerCount: 1,
          },
        ],
        currentDocument: { ...mockDoc, selectedLayerId: "layer-1" },
        pendingTargetLayerId: "layer-1",
      });

      useDocumentStore.getState().resetForUserSwitch();

      const state = useDocumentStore.getState();
      expect(state.documents).toEqual([]);
      expect(state.currentDocument).toBeNull();
      expect(state.pendingTargetLayerId).toBeNull();
    });
  });
});
