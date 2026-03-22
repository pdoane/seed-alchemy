import {
  Dialog,
  Flex,
  Text,
  Button,
  Select,
  Badge,
  Spinner,
  Box,
  Progress,
  IconButton,
} from "@radix-ui/themes";
import { useState, useCallback, useEffect, useRef } from "react";
import type { ModelFolder } from "../../shared/types/models";
import { api } from "../api/client";
import { UploadSimpleIcon, XIcon } from "@phosphor-icons/react";

const FOLDER_OPTIONS: { value: ModelFolder; label: string }[] = [
  { value: "checkpoints", label: "Checkpoints" },
  { value: "loras", label: "LoRAs" },
  { value: "vae", label: "VAE" },
  { value: "controlnet", label: "ControlNet" },
  { value: "unet", label: "UNet" },
  { value: "clip", label: "CLIP" },
  { value: "embeddings", label: "Embeddings" },
  { value: "upscale_models", label: "Upscalers" },
];

const MODEL_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"];

interface ImportModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImportSuccess: () => void;
}

interface FileEntry {
  file: File;
  folder: ModelFolder;
}

// Suggest folder based on filename and size
function suggestFolder(filename: string, fileSize: number): ModelFolder {
  const lower = filename.toLowerCase();
  const sizeGB = fileSize / 1_000_000_000;

  if (
    lower.includes("lora") ||
    lower.includes("loha") ||
    lower.includes("lokr")
  )
    return "loras";
  if (lower.includes("vae")) return "vae";
  if (lower.includes("controlnet") || lower.includes("control_"))
    return "controlnet";
  if (
    lower.includes("upscale") ||
    lower.includes("esrgan") ||
    lower.includes("realesrgan")
  )
    return "upscale_models";
  if (lower.includes("embed") || lower.includes("ti_")) return "embeddings";
  if (lower.includes("clip") && !lower.includes("clip_skip")) return "clip";
  if (lower.includes("unet") || lower.includes("diffusion_model"))
    return "unet";
  if (sizeGB > 1.5) return "checkpoints";
  if (sizeGB < 0.5) return "loras";
  return "checkpoints";
}

function formatSize(bytes: number): string {
  const gb = bytes / 1_000_000_000;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / 1_000_000;
  return `${mb.toFixed(1)} MB`;
}

function isValidModelFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return MODEL_EXTENSIONS.some((e) => lower.endsWith(e));
}

export function ImportModelDialog({
  open,
  onOpenChange,
  onImportSuccess,
}: ImportModelDialogProps) {
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadIndex, setUploadIndex] = useState(0);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setEntries([]);
      setUploadProgress(null);
      setUploadIndex(0);
      setError(null);
      setIsDragOver(false);
      setUploading(false);
    }
  }, [open]);

  const addFiles = useCallback(
    (files: File[]) => {
      setError(null);

      const invalid = files.filter((f) => !isValidModelFile(f.name));
      if (invalid.length > 0) {
        setError(
          `Skipped ${invalid.length} unsupported file(s). Expected: ${MODEL_EXTENSIONS.join(", ")}`
        );
      }

      const valid = files.filter((f) => isValidModelFile(f.name));
      if (valid.length === 0) return;

      // Deduplicate by name against existing entries
      const existingNames = new Set(entries.map((e) => e.file.name));
      const newEntries = valid
        .filter((f) => !existingNames.has(f.name))
        .map((f) => ({
          file: f,
          folder: suggestFolder(f.name, f.size),
        }));

      setEntries((prev) => [...prev, ...newEntries]);
    },
    [entries]
  );

  const removeEntry = useCallback((index: number) => {
    setEntries((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const updateFolder = useCallback((index: number, folder: ModelFolder) => {
    setEntries((prev) =>
      prev.map((entry, i) => (i === index ? { ...entry, folder } : entry))
    );
  }, []);

  const handleFileSelect = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (files && files.length > 0) {
        addFiles(Array.from(files));
      }
      event.target.value = "";
    },
    [addFiles]
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
      setIsDragOver(false);

      const files = event.dataTransfer.files;
      if (files.length > 0) {
        addFiles(Array.from(files));
      }
    },
    [addFiles]
  );

  const handleUpload = useCallback(async () => {
    if (entries.length === 0) return;

    setUploading(true);
    setError(null);

    let anySuccess = false;
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]!;
      setUploadIndex(i);
      setUploadProgress(0);

      try {
        await api.uploadModel(entry.file, entry.folder, (progress) => {
          setUploadProgress(progress);
        });
        anySuccess = true;
      } catch (err) {
        setError(
          `Failed to upload ${entry.file.name}: ${err instanceof Error ? err.message : "Unknown error"}`
        );
        setUploading(false);
        setUploadProgress(null);
        if (anySuccess) onImportSuccess();
        return;
      }
    }

    setUploading(false);
    setUploadProgress(null);
    onImportSuccess();
    onOpenChange(false);
  }, [entries, onImportSuccess, onOpenChange]);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="500px">
        <Dialog.Title>Import Models</Dialog.Title>
        <Dialog.Description size="2" color="gray" mb="4">
          Upload model files to your models folder.
        </Dialog.Description>

        <Flex direction="column" gap="4">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".safetensors,.ckpt,.pt,.pth,.bin"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />

          {/* Drag & drop zone */}
          <Box
            p="4"
            className={`cursor-pointer rounded border-2 border-dashed transition-colors ${
              isDragOver
                ? "border-[var(--accent-9)] bg-[var(--accent-3)]"
                : "border-[var(--gray-6)] bg-[var(--gray-2)] hover:border-[var(--gray-8)]"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Flex direction="column" align="center" gap="2">
              <UploadSimpleIcon size={24} weight="bold" color="var(--gray-9)" />
              <Text size="2" color="gray" align="center">
                Drop model files here, or click to browse
              </Text>
              <Text size="1" color="gray">
                .safetensors, .ckpt, .pt, .pth, .bin
              </Text>
            </Flex>
          </Box>

          {/* File list */}
          {entries.length > 0 && (
            <Flex
              direction="column"
              gap="2"
              style={{ maxHeight: 300, overflowY: "auto" }}
            >
              {entries.map((entry, index) => (
                <Box
                  key={entry.file.name}
                  p="3"
                  className="rounded border border-[var(--gray-6)]"
                  style={{ backgroundColor: "var(--gray-2)" }}
                >
                  <Flex direction="column" gap="2">
                    <Flex justify="between" align="center" gap="2">
                      <Text
                        size="2"
                        weight="medium"
                        style={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          minWidth: 0,
                          flex: 1,
                        }}
                      >
                        {entry.file.name}
                      </Text>
                      <Flex align="center" gap="2" flexShrink="0">
                        <Badge color="gray">
                          {formatSize(entry.file.size)}
                        </Badge>
                        <IconButton
                          size="1"
                          variant="ghost"
                          color="gray"
                          disabled={uploading}
                          onClick={() => removeEntry(index)}
                        >
                          <XIcon size={14} />
                        </IconButton>
                      </Flex>
                    </Flex>
                    <Select.Root
                      size="1"
                      value={entry.folder}
                      disabled={uploading}
                      onValueChange={(v) =>
                        updateFolder(index, v as ModelFolder)
                      }
                    >
                      <Select.Trigger />
                      <Select.Content>
                        {FOLDER_OPTIONS.map((opt) => (
                          <Select.Item key={opt.value} value={opt.value}>
                            {opt.label}
                          </Select.Item>
                        ))}
                      </Select.Content>
                    </Select.Root>
                  </Flex>
                </Box>
              ))}
            </Flex>
          )}

          {/* Upload progress */}
          {uploading && uploadProgress !== null && (
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                Uploading {uploadIndex + 1} of {entries.length}...{" "}
                {Math.round(uploadProgress)}%
              </Text>
              <Progress value={uploadProgress} />
            </Flex>
          )}

          {/* Error message */}
          {error && (
            <Text size="2" color="red">
              {error}
            </Text>
          )}

          {/* Actions */}
          <Flex gap="3" justify="end" mt="2">
            <Dialog.Close>
              <Button variant="soft" color="gray" disabled={uploading}>
                Cancel
              </Button>
            </Dialog.Close>
            <Button
              onClick={handleUpload}
              disabled={entries.length === 0 || uploading}
            >
              {uploading ? (
                <>
                  <Spinner size="1" />
                  Uploading...
                </>
              ) : entries.length <= 1 ? (
                "Upload"
              ) : (
                `Upload ${entries.length} Files`
              )}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
