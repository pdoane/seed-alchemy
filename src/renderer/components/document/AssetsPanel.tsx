import { type ReactNode } from "react";
import { Box, Flex, Text } from "@radix-ui/themes";
import {
  ArrowsOutIcon,
  ImageSquareIcon,
  SparkleIcon,
  StackIcon,
  ScanIcon,
  UserFocusIcon,
} from "@phosphor-icons/react";
import type { Asset } from "../../../shared/types/document";
import type { OperationRecord } from "../../../shared/types/image";
import { formatPreprocessorName } from "../../../shared/utils";

interface AssetsPanelProps {
  assets: Asset[];
  getAssetUrl?: (filename: string) => string;
  onAssetDragStart?: (assetId: string) => void;
  onAssetDoubleClick?: (assetId: string) => void;
}

// Get icon based on operation type
function getOperationIcon(operation: OperationRecord): ReactNode {
  switch (operation.type) {
    case "raw":
      return <ImageSquareIcon size={16} weight="fill" />;
    case "generate":
      return <SparkleIcon size={16} weight="fill" />;
    case "detect":
      return <ScanIcon size={16} weight="fill" />;
    case "flatten":
      return <StackIcon size={16} weight="fill" />;
    case "enhance":
      return operation.params.enhanceType === "face_detailer" ? (
        <UserFocusIcon size={16} weight="fill" />
      ) : (
        <ArrowsOutIcon size={16} weight="fill" />
      );
    case "sequence": {
      // Show icon of last operation in sequence
      const lastOp = operation.operations[operation.operations.length - 1];
      return lastOp ? (
        getOperationIcon(lastOp)
      ) : (
        <SparkleIcon size={16} weight="fill" />
      );
    }
  }
}

// Get display label for operation
function getOperationLabel(operation: OperationRecord): string {
  switch (operation.type) {
    case "raw":
      return "Raw";
    case "generate":
      return "Generated";
    case "detect":
      return formatPreprocessorName(operation.params.detector);
    case "flatten":
      return "Flattened";
    case "enhance":
      return operation.params.enhanceType === "face_detailer"
        ? "Face Detail"
        : "Upscaled";
    case "sequence": {
      // Show label of last operation in sequence
      const lastOp = operation.operations[operation.operations.length - 1];
      return lastOp ? getOperationLabel(lastOp) : "Sequence";
    }
  }
}

export function AssetsPanel({
  assets,
  getAssetUrl,
  onAssetDragStart,
  onAssetDoubleClick,
}: AssetsPanelProps) {
  const handleDragStart = (e: React.DragEvent, assetId: string) => {
    e.dataTransfer.setData("application/x-asset-id", assetId);
    e.dataTransfer.effectAllowed = "copy";
    onAssetDragStart?.(assetId);
  };

  return (
    <Flex direction="column" className="h-full bg-[var(--gray-2)]">
      <Flex
        align="center"
        px="3"
        py="2"
        className="border-b border-[var(--gray-6)]"
      >
        <Text size="2" weight="medium">
          Assets
        </Text>
        {assets.length > 0 && (
          <Text size="1" color="gray" className="ml-2">
            ({assets.length})
          </Text>
        )}
      </Flex>

      <Box className="flex-1 overflow-y-auto" p="2">
        {assets.length === 0 ? (
          <Flex
            align="center"
            justify="center"
            className="h-full text-[var(--gray-9)]"
          >
            <Text size="1">No assets</Text>
          </Flex>
        ) : (
          <Flex wrap="wrap" gap="2">
            {assets.map((asset) => {
              const thumbnailUrl = getAssetUrl?.(asset.filename);
              return (
                <Flex
                  key={asset.filename}
                  direction="column"
                  align="center"
                  gap="1"
                  p="2"
                  className="cursor-grab rounded transition-colors hover:bg-[var(--gray-4)] active:cursor-grabbing"
                  title={`${getOperationLabel(asset.metadata.operation)}\n${asset.filename}\nDrag to viewport or double-click to add as layer`}
                  draggable
                  onDragStart={(e) => handleDragStart(e, asset.filename)}
                  onDoubleClick={() => onAssetDoubleClick?.(asset.filename)}
                >
                  <Box className="relative flex h-12 w-12 items-center justify-center overflow-hidden rounded bg-[var(--gray-5)]">
                    {thumbnailUrl ? (
                      <img
                        src={thumbnailUrl}
                        alt=""
                        className="h-full w-full object-cover"
                        draggable={false}
                      />
                    ) : (
                      getOperationIcon(asset.metadata.operation)
                    )}
                  </Box>
                  <Text size="1" className="max-w-[60px] truncate">
                    {getOperationLabel(asset.metadata.operation)}
                  </Text>
                </Flex>
              );
            })}
          </Flex>
        )}
      </Box>
    </Flex>
  );
}
