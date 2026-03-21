import {
  Box,
  Flex,
  Text,
  Badge,
  ScrollArea,
  Spinner,
  Button,
} from "@radix-ui/themes";
import { useEffect, useState, useCallback } from "react";
import type { ModelInfo, ModelFolder } from "../../../shared/types/models";
import { api } from "../../api/client";
import { ImportModelDialog } from "../ImportModelDialog";

// Folder display info
const FOLDER_INFO: Record<ModelFolder, { label: string; icon: string }> = {
  checkpoints: { label: "Checkpoints", icon: "🎯" },
  diffusion_models: { label: "Diffusion Models", icon: "⚡" },
  loras: { label: "LoRAs", icon: "🔗" },
  vae: { label: "VAE", icon: "🎨" },
  controlnet: { label: "ControlNet", icon: "🎮" },
  unet: { label: "UNet", icon: "🧠" },
  clip: { label: "CLIP", icon: "📝" },
  text_encoders: { label: "Text Encoders", icon: "🔤" },
  clip_vision: { label: "CLIP Vision", icon: "👁️" },
  embeddings: { label: "Embeddings", icon: "💎" },
  upscale_models: { label: "Upscalers", icon: "🔍" },
  ipadapter: { label: "IP-Adapter", icon: "🖼️" },
};

interface FolderCount {
  folder: ModelFolder;
  count: number;
}

export function ModelsMode() {
  const [folders, setFolders] = useState<FolderCount[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<ModelFolder | null>(
    null
  );
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);

  // Load available folders
  const loadFolders = useCallback(async () => {
    try {
      const data = await api.get<FolderCount[]>("/api/models/folders");
      setFolders(data);
      // Auto-select first folder if available
      const firstFolder = data[0];
      if (firstFolder) {
        setSelectedFolder((current) => current ?? firstFolder.folder);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load folders");
    } finally {
      setLoading(false);
    }
  }, []);

  // Load folders on mount
  useEffect(() => {
    loadFolders();
  }, [loadFolders]);

  // Load models for current folder
  const loadModels = useCallback(async () => {
    if (!selectedFolder) return;
    setLoading(true);
    setSelectedModel(null);
    try {
      const data = await api.get<ModelInfo[]>(
        `/api/models?folder=${selectedFolder}`
      );
      setModels(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, [selectedFolder]);

  // Load models when folder changes
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Handle import success - reload folders and models
  const handleImportSuccess = useCallback(async () => {
    await loadFolders();
    await loadModels();
  }, [loadFolders, loadModels]);

  return (
    <>
      <Flex className="h-full">
        {/* Folder Sidebar */}
        <FolderSidebar
          folders={folders}
          selectedFolder={selectedFolder}
          onSelectFolder={setSelectedFolder}
          onImportClick={() => setImportDialogOpen(true)}
        />

        {/* Main Content */}
        <Flex direction="column" className="flex-1 overflow-hidden">
          {loading ? (
            <Flex align="center" justify="center" className="h-full">
              <Spinner size="3" />
            </Flex>
          ) : error ? (
            <Flex align="center" justify="center" className="h-full">
              <Text color="red">{error}</Text>
            </Flex>
          ) : (
            <>
              <Box className="flex-1 overflow-hidden">
                <ModelGrid
                  models={models}
                  selectedModel={selectedModel}
                  onSelectModel={setSelectedModel}
                />
              </Box>
              {selectedModel && (
                <ModelDetailPanel
                  model={selectedModel}
                  onClose={() => setSelectedModel(null)}
                  onDelete={async () => {
                    setSelectedModel(null);
                    await loadFolders();
                    await loadModels();
                  }}
                />
              )}
            </>
          )}
        </Flex>
      </Flex>

      <ImportModelDialog
        open={importDialogOpen}
        onOpenChange={setImportDialogOpen}
        onImportSuccess={handleImportSuccess}
      />
    </>
  );
}

interface FolderSidebarProps {
  folders: FolderCount[];
  selectedFolder: ModelFolder | null;
  onSelectFolder: (folder: ModelFolder) => void;
  onImportClick: () => void;
}

function FolderSidebar({
  folders,
  selectedFolder,
  onSelectFolder,
  onImportClick,
}: FolderSidebarProps) {
  return (
    <Box
      className="w-48 border-r border-[var(--gray-6)]"
      style={{ backgroundColor: "var(--gray-2)" }}
    >
      <Box p="3" className="border-b border-[var(--gray-6)]">
        <Flex justify="between" align="center">
          <Text size="2" weight="bold">
            Model Folders
          </Text>
          <Button size="1" onClick={onImportClick}>
            Import
          </Button>
        </Flex>
      </Box>
      <ScrollArea>
        <Flex direction="column" gap="1" p="2">
          {folders.map(({ folder, count }) => {
            const info = FOLDER_INFO[folder];
            const isSelected = folder === selectedFolder;
            return (
              <Box
                key={folder}
                p="2"
                className={`cursor-pointer rounded ${
                  isSelected
                    ? "bg-[var(--violet-9)]"
                    : "hover:bg-[var(--gray-4)]"
                }`}
                onClick={() => onSelectFolder(folder)}
              >
                <Flex align="center" justify="between">
                  <Flex align="center" gap="2">
                    <span>{info.icon}</span>
                    <Text size="2">{info.label}</Text>
                  </Flex>
                  <Badge size="1" color="gray">
                    {count}
                  </Badge>
                </Flex>
              </Box>
            );
          })}
          {folders.length === 0 && (
            <Text size="2" color="gray" align="center" className="py-4">
              No model folders found
            </Text>
          )}
        </Flex>
      </ScrollArea>
    </Box>
  );
}

interface ModelGridProps {
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  onSelectModel: (model: ModelInfo) => void;
}

function ModelGrid({ models, selectedModel, onSelectModel }: ModelGridProps) {
  if (models.length === 0) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text color="gray">No models in this folder</Text>
      </Flex>
    );
  }

  return (
    <ScrollArea className="h-full">
      <Box p="4">
        <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-4">
          {models.map((model) => (
            <ModelCard
              key={model.path}
              model={model}
              isSelected={selectedModel?.path === model.path}
              onSelect={() => onSelectModel(model)}
            />
          ))}
        </div>
      </Box>
    </ScrollArea>
  );
}

interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  onSelect: () => void;
}

function ModelCard({ model, isSelected, onSelect }: ModelCardProps) {
  const sizeGB = (model.fileSizeBytes / 1_000_000_000).toFixed(1);

  return (
    <Box
      className={`cursor-pointer rounded-lg border transition-colors ${
        isSelected
          ? "border-[var(--violet-9)] ring-2 ring-[var(--violet-9)]"
          : "border-[var(--gray-6)] hover:border-[var(--violet-8)]"
      }`}
      style={{ backgroundColor: "var(--gray-3)" }}
      onClick={onSelect}
    >
      {/* Thumbnail */}
      <Box
        className="flex h-32 items-center justify-center rounded-t-lg"
        style={{ backgroundColor: "var(--gray-4)" }}
      >
        {model.thumbnail ? (
          <img
            src={model.thumbnail}
            alt={model.filename}
            className="h-full w-full rounded-t-lg object-contain"
          />
        ) : (
          <Text size="6" color="gray">
            📦
          </Text>
        )}
      </Box>

      {/* Model info */}
      <Box p="3">
        <Text
          size="2"
          weight="medium"
          className="block truncate"
          title={model.filename}
        >
          {model.title || model.filename}
        </Text>

        {model.title && (
          <Text size="1" color="gray" className="block truncate">
            {model.filename}
          </Text>
        )}

        <Flex gap="2" mt="2" wrap="wrap">
          {model.architecture !== "unknown" && (
            <Badge size="1" color="violet">
              {model.architecture.toUpperCase()}
            </Badge>
          )}
          <Badge size="1" color="gray">
            {sizeGB} GB
          </Badge>
        </Flex>
      </Box>
    </Box>
  );
}

interface ModelDetailPanelProps {
  model: ModelInfo;
  onClose: () => void;
  onDelete: () => void;
}

function ModelDetailPanel({ model, onClose, onDelete }: ModelDetailPanelProps) {
  const [deleting, setDeleting] = useState(false);
  const sizeGB = (model.fileSizeBytes / 1_000_000_000).toFixed(2);

  const handleDelete = async () => {
    if (!confirm(`Delete "${model.filename}"? This cannot be undone.`)) {
      return;
    }
    setDeleting(true);
    try {
      await api.deleteModel(model.folder, model.filename);
      onDelete();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Box
      className="border-t border-[var(--gray-6)]"
      style={{ backgroundColor: "var(--gray-2)" }}
    >
      <ScrollArea style={{ maxHeight: "300px" }}>
        <Flex p="4" gap="4">
          {/* Thumbnail */}
          {model.thumbnail && (
            <Box className="flex-shrink-0">
              <img
                src={model.thumbnail}
                alt={model.filename}
                className="h-48 w-48 rounded-lg object-contain"
              />
            </Box>
          )}

          {/* Details */}
          <Flex direction="column" gap="3" className="min-w-0 flex-1">
            {/* Header with close button */}
            <Flex justify="between" align="start">
              <Box>
                <Text size="5" weight="bold">
                  {model.title || model.filename}
                </Text>
                {model.title && (
                  <Text size="2" color="gray" className="block">
                    {model.filename}
                  </Text>
                )}
              </Box>
              <Box
                className="cursor-pointer rounded p-1 hover:bg-[var(--gray-4)]"
                onClick={onClose}
              >
                <Text size="3">✕</Text>
              </Box>
            </Flex>

            {/* Metadata row */}
            <Flex gap="4" wrap="wrap">
              {model.architecture !== "unknown" && (
                <Flex direction="column">
                  <Text size="1" color="gray">
                    Architecture
                  </Text>
                  <Text size="2">{model.architecture.toUpperCase()}</Text>
                </Flex>
              )}
              {model.author && (
                <Flex direction="column">
                  <Text size="1" color="gray">
                    Author
                  </Text>
                  <Text size="2">{model.author}</Text>
                </Flex>
              )}
              <Flex direction="column">
                <Text size="1" color="gray">
                  Size
                </Text>
                <Text size="2">{sizeGB} GB</Text>
              </Flex>
              {model.tensorCount > 0 && (
                <Flex direction="column">
                  <Text size="1" color="gray">
                    Tensors
                  </Text>
                  <Text size="2">{model.tensorCount.toLocaleString()}</Text>
                </Flex>
              )}
              {model.license && (
                <Flex direction="column">
                  <Text size="1" color="gray">
                    License
                  </Text>
                  <Text size="2">{model.license}</Text>
                </Flex>
              )}
            </Flex>

            {/* Description */}
            {model.description && (
              <Box>
                <Text size="1" color="gray" className="mb-1 block">
                  Description
                </Text>
                <Text size="2" className="block">
                  {model.description}
                </Text>
              </Box>
            )}

            {/* Path */}
            <Box>
              <Text size="1" color="gray" className="mb-1 block">
                Path
              </Text>
              <Text size="1" className="block font-mono" color="gray">
                {model.path}
              </Text>
            </Box>

            {/* Actions */}
            <Flex gap="2" mt="2">
              <Button
                color="red"
                size="1"
                onClick={handleDelete}
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Delete"}
              </Button>
            </Flex>
          </Flex>
        </Flex>
      </ScrollArea>
    </Box>
  );
}
