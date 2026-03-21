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
} from "@radix-ui/themes";
import { useState, useCallback, useEffect, useRef } from "react";
import type { ModelFolder } from "../../shared/types/models";
import { api } from "../api/client";
import { UploadSimpleIcon } from "@phosphor-icons/react";

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

export function ImportModelDialog({
  open,
  onOpenChange,
  onImportSuccess,
}: ImportModelDialogProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetFolder, setTargetFolder] = useState<ModelFolder>("checkpoints");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setSelectedFile(null);
      setTargetFolder("checkpoints");
      setUploadProgress(null);
      setError(null);
      setIsDragOver(false);
      setUploading(false);
    }
  }, [open]);

  const formatSize = (bytes: number): string => {
    const gb = bytes / 1_000_000_000;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / 1_000_000;
    return `${mb.toFixed(1)} MB`;
  };

  const handleFile = useCallback((file: File) => {
    setError(null);
    setSelectedFile(null);

    // Validate file extension
    const ext = file.name.toLowerCase();
    if (!MODEL_EXTENSIONS.some((e) => ext.endsWith(e))) {
      setError(
        `Unsupported file type. Expected: ${MODEL_EXTENSIONS.join(", ")}`
      );
      return;
    }

    setSelectedFile(file);
    setTargetFolder(suggestFolder(file.name, file.size));
  }, []);

  const handleFileSelect = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        handleFile(file);
      }
      event.target.value = "";
    },
    [handleFile]
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

      const file = event.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      await api.uploadModel(selectedFile, targetFolder, (progress) => {
        setUploadProgress(progress);
      });
      onImportSuccess();
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload model");
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  }, [selectedFile, targetFolder, onImportSuccess, onOpenChange]);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="500px">
        <Dialog.Title>Import Model</Dialog.Title>
        <Dialog.Description size="2" color="gray" mb="4">
          Upload a model file to your models folder.
        </Dialog.Description>

        <Flex direction="column" gap="4">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
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
                Drop a model file here, or click to browse
              </Text>
              <Text size="1" color="gray">
                .safetensors, .ckpt, .pt, .pth, .bin
              </Text>
            </Flex>
          </Box>

          {/* Selected file preview */}
          {selectedFile && (
            <Box
              p="3"
              className="rounded border border-[var(--gray-6)]"
              style={{ backgroundColor: "var(--gray-2)" }}
            >
              <Flex direction="column" gap="3">
                <Flex justify="between" align="center">
                  <Text size="2" weight="medium">
                    {selectedFile.name}
                  </Text>
                  <Badge color="gray">{formatSize(selectedFile.size)}</Badge>
                </Flex>
              </Flex>
            </Box>
          )}

          {/* Target folder selection */}
          {selectedFile && (
            <Flex direction="column" gap="2">
              <Text size="2" weight="medium">
                Target Folder
              </Text>
              <Select.Root
                value={targetFolder}
                onValueChange={(v) => setTargetFolder(v as ModelFolder)}
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
          )}

          {/* Upload progress */}
          {uploadProgress !== null && (
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                Uploading... {Math.round(uploadProgress)}%
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
              disabled={!selectedFile || uploading}
            >
              {uploading ? (
                <>
                  <Spinner size="1" />
                  Uploading...
                </>
              ) : (
                "Upload"
              )}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
