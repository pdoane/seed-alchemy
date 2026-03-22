import {
  Dialog,
  Flex,
  Text,
  Button,
  Badge,
  Spinner,
  Box,
  Progress,
  IconButton,
} from "@radix-ui/themes";
import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../api/client";
import { UploadSimpleIcon, XIcon } from "@phosphor-icons/react";

const MODEL_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"];

interface ImportModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImportSuccess: () => void;
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
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadIndex, setUploadIndex] = useState(0);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setFiles([]);
      setUploadProgress(null);
      setUploadIndex(0);
      setError(null);
      setIsDragOver(false);
      setUploading(false);
    }
  }, [open]);

  const addFiles = useCallback(
    (newFiles: File[]) => {
      setError(null);

      const invalid = newFiles.filter((f) => !isValidModelFile(f.name));
      if (invalid.length > 0) {
        setError(
          `Skipped ${invalid.length} unsupported file(s). Expected: ${MODEL_EXTENSIONS.join(", ")}`
        );
      }

      const valid = newFiles.filter((f) => isValidModelFile(f.name));
      if (valid.length === 0) return;

      // Deduplicate by name against existing files
      const existingNames = new Set(files.map((f) => f.name));
      const deduped = valid.filter((f) => !existingNames.has(f.name));

      setFiles((prev) => [...prev, ...deduped]);
    },
    [files]
  );

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleFileSelect = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files;
      if (selected && selected.length > 0) {
        addFiles(Array.from(selected));
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

      const dropped = event.dataTransfer.files;
      if (dropped.length > 0) {
        addFiles(Array.from(dropped));
      }
    },
    [addFiles]
  );

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;

    setUploading(true);
    setError(null);

    let anySuccess = false;
    for (let i = 0; i < files.length; i++) {
      const file = files[i]!;
      setUploadIndex(i);
      setUploadProgress(0);

      try {
        await api.uploadModel(file, (progress) => {
          setUploadProgress(progress);
        });
        anySuccess = true;
      } catch (err) {
        setError(
          `Failed to upload ${file.name}: ${err instanceof Error ? err.message : "Unknown error"}`
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
  }, [files, onImportSuccess, onOpenChange]);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="500px">
        <Dialog.Title>Import Models</Dialog.Title>
        <Dialog.Description size="2" color="gray" mb="4">
          Upload model files. The server will automatically classify each file
          into the correct folder.
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
          {files.length > 0 && (
            <Flex
              direction="column"
              gap="2"
              style={{ maxHeight: 300, overflowY: "auto" }}
            >
              {files.map((file, index) => (
                <Box
                  key={file.name}
                  p="3"
                  className="rounded border border-[var(--gray-6)]"
                  style={{ backgroundColor: "var(--gray-2)" }}
                >
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
                      {file.name}
                    </Text>
                    <Flex align="center" gap="2" flexShrink="0">
                      <Badge color="gray">{formatSize(file.size)}</Badge>
                      <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        disabled={uploading}
                        onClick={() => removeFile(index)}
                      >
                        <XIcon size={14} />
                      </IconButton>
                    </Flex>
                  </Flex>
                </Box>
              ))}
            </Flex>
          )}

          {/* Upload progress */}
          {uploading && uploadProgress !== null && (
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                Uploading {uploadIndex + 1} of {files.length}...{" "}
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
              disabled={files.length === 0 || uploading}
            >
              {uploading ? (
                <>
                  <Spinner size="1" />
                  Uploading...
                </>
              ) : files.length <= 1 ? (
                "Upload"
              ) : (
                `Upload ${files.length} Files`
              )}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
