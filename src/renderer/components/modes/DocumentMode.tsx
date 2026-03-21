import { useEffect } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { ToolType } from "../../../shared/types/document";
import { ToolPalette } from "../document/ToolPalette";
import { LayerPanel } from "../document/LayerPanel";
import { AssetsPanel } from "../document/AssetsPanel";
import { DocumentBrowser } from "../document/DocumentBrowser";
import { DocumentViewport } from "../document/DocumentViewport";
import { ParameterPanel } from "../ParameterPanel";
import { useDocumentStore } from "../../store/documentStore";
import { useImageStore } from "../../store/imageStore";
import { api } from "../../api/client";

export function DocumentMode() {
  // Document store state
  const documents = useDocumentStore((s) => s.documents);
  const loadDocuments = useDocumentStore((s) => s.loadDocuments);
  const currentDocument = useDocumentStore((s) => s.currentDocument);
  const loadDocument = useDocumentStore((s) => s.loadDocument);
  const selectedLayerId = useDocumentStore(
    (s) => s.currentDocument?.selectedLayerId ?? null
  );
  const setSelectedLayerId = useDocumentStore((s) => s.setSelectedLayerId);
  const toggleLayerVisibility = useDocumentStore(
    (s) => s.toggleLayerVisibility
  );
  const addLayer = useDocumentStore((s) => s.addLayer);
  const setLayerOpacity = useDocumentStore((s) => s.setLayerOpacity);
  const setLayerBlendMode = useDocumentStore((s) => s.setLayerBlendMode);
  const setLayerPosition = useDocumentStore((s) => s.setLayerPosition);
  const updateLayer = useDocumentStore((s) => s.updateLayer);
  const removeLayer = useDocumentStore((s) => s.removeLayer);
  const activeTool = useDocumentStore((s) => s.documentUi.activeTool);
  const setActiveTool = (tool: ToolType) =>
    useDocumentStore.getState().setDocumentUi({ activeTool: tool });
  const createDocument = useDocumentStore((s) => s.createDocument);
  const setPendingTargetLayerId = useDocumentStore(
    (s) => s.setPendingTargetLayerId
  );

  // Image store - for generation
  const generate = useImageStore((s) => s.generate);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if ((e.key === "Delete" || e.key === "Backspace") && selectedLayerId) {
        e.preventDefault();
        removeLayer(selectedLayerId);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedLayerId, removeLayer]);

  // Handlers
  const handleSelectDocument = (docId: string) => {
    loadDocument(docId);
  };

  const handleToggleVisibility = (layerId: string) => {
    toggleLayerVisibility(layerId);
  };

  const handleAddLayer = () => {
    if (!currentDocument) return;
    const layerCount = currentDocument.layers.length;
    addLayer(`Layer ${layerCount + 1}`);
  };

  const handleCreateDocument = () => {
    createDocument();
  };

  // Override generate to pass documentId and capture target layer
  const handleGenerate = () => {
    if (currentDocument) {
      setPendingTargetLayerId(selectedLayerId);
      generate(currentDocument.id);
    }
  };

  // Helper to get asset URL
  const getAssetUrl = (filename: string) => {
    if (!currentDocument) return "";
    return api.getDocumentAssetUrl(currentDocument.id, filename);
  };

  // Create layer from asset (for drag-drop or double-click)
  const handleAddLayerFromAsset = (filename: string, x = 0, y = 0) => {
    if (!currentDocument) return;
    const asset = currentDocument.assets.find((a) => a.filename === filename);
    if (!asset) return;

    const layerCount = currentDocument.layers.length;
    addLayer(`Layer ${layerCount + 1}`, filename);

    // Set position if not at origin
    if (x !== 0 || y !== 0) {
      // Get the newly created layer (it will be the last one)
      const newLayerId = currentDocument.layers[currentDocument.layers.length];
      if (newLayerId) {
        setLayerPosition(newLayerId.id, x, y);
      }
    }
  };

  return (
    <PanelGroup
      direction="horizontal"
      className="h-full"
      autoSaveId="document-mode"
    >
      {/* Left + Center: Parameters, Viewport, Documents */}
      <Panel minSize={50}>
        <PanelGroup direction="vertical" autoSaveId="document-mode-main">
          {/* Top row: Parameters + Viewport */}
          <Panel minSize={40}>
            <PanelGroup direction="horizontal" autoSaveId="document-mode-top">
              <Panel defaultSize={25} minSize={15} maxSize={40}>
                <ParameterPanel
                  documentId={currentDocument?.id}
                  onGenerate={handleGenerate}
                />
              </Panel>
              <PanelResizeHandle className="w-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />
              <Panel minSize={30}>
                <div className="flex h-full">
                  <ToolPalette
                    activeTool={activeTool}
                    onToolChange={setActiveTool}
                  />
                  <div className="flex-1">
                    <DocumentViewport
                      activeTool={activeTool}
                      canvasSize={currentDocument?.canvasSize}
                      layers={currentDocument?.layers}
                      assets={currentDocument?.assets}
                      getAssetUrl={getAssetUrl}
                      selectedLayerId={selectedLayerId}
                      onLayerSelect={setSelectedLayerId}
                      onAssetDrop={handleAddLayerFromAsset}
                    />
                  </div>
                </div>
              </Panel>
            </PanelGroup>
          </Panel>
          <PanelResizeHandle className="h-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />
          {/* Bottom: Documents browser (spans full width up to Layers/Assets) */}
          <Panel defaultSize={20} minSize={10} maxSize={40}>
            <DocumentBrowser
              documents={documents}
              selectedDocumentId={currentDocument?.id ?? null}
              onSelectDocument={handleSelectDocument}
              onCreateDocument={handleCreateDocument}
            />
          </Panel>
        </PanelGroup>
      </Panel>

      <PanelResizeHandle className="w-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />

      {/* Right side: Layers + Assets (full height) */}
      <Panel defaultSize={15} minSize={12} maxSize={30}>
        <PanelGroup direction="vertical" autoSaveId="document-mode-right">
          <Panel defaultSize={60} minSize={20}>
            <LayerPanel
              layers={currentDocument?.layers ?? []}
              assets={currentDocument?.assets ?? []}
              selectedLayerId={selectedLayerId}
              onSelectLayer={setSelectedLayerId}
              onToggleVisibility={handleToggleVisibility}
              onAddLayer={handleAddLayer}
              onRemoveLayer={removeLayer}
              getAssetUrl={getAssetUrl}
              onLayerOpacityChange={setLayerOpacity}
              onLayerBlendModeChange={setLayerBlendMode}
              onLayerNameChange={(layerId, name) =>
                updateLayer(layerId, { name })
              }
              onLayerPositionChange={setLayerPosition}
            />
          </Panel>
          <PanelResizeHandle className="h-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />
          <Panel defaultSize={40} minSize={15} collapsible>
            <AssetsPanel
              assets={currentDocument?.assets ?? []}
              getAssetUrl={getAssetUrl}
              onAssetDoubleClick={(assetId) => handleAddLayerFromAsset(assetId)}
            />
          </Panel>
        </PanelGroup>
      </Panel>
    </PanelGroup>
  );
}
