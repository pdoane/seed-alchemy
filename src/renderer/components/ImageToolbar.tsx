import { Box, DropdownMenu, Flex } from "@radix-ui/themes";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowsOutIcon,
  ImageIcon,
  ImageSquareIcon,
  QuotesIcon,
  AcornIcon,
  RulerIcon,
  AsteriskIcon,
  EyeIcon,
  InfoIcon,
  TrashIcon,
  SlidersHorizontalIcon,
  UserFocusIcon,
} from "@phosphor-icons/react";
import { ToolbarButton } from "./ToolbarButton";
import {
  useImageStore,
  selectSelectedImage,
  selectCanNavigateBack,
  selectCanNavigateForward,
} from "../store/imageStore";
import type { ImageMetadata, ImageParams } from "../../shared/types/image";
import type { ControlNetConfig } from "../../shared/types/controlnet";

interface ImageToolbarProps {
  showMetadata: boolean;
  setShowMetadata: (show: boolean) => void;
  onDelete: () => void;
  metadata: ImageMetadata | null;
  onUseParams: (params: Partial<ImageParams>, label: string) => void;
  onUseAsSource: () => void;
  onUseAsReference: () => void;
  onUseAsControlNet: (index: number) => void;
  onUseAspectRatio: () => void;
  controlNets: ControlNetConfig[];
  isGenerating: boolean;
  onFaceDetailer: () => void;
  onUpscale: () => void;
}

export function ImageToolbar({
  showMetadata,
  setShowMetadata,
  onDelete,
  metadata,
  onUseParams,
  onUseAsSource,
  onUseAsReference,
  onUseAsControlNet,
  onUseAspectRatio,
  controlNets,
  isGenerating,
  onFaceDetailer,
  onUpscale,
}: ImageToolbarProps) {
  // Extract params if this is a generated image
  const params =
    metadata?.operation.type === "generate"
      ? metadata.operation.params
      : undefined;

  const selectedImage = useImageStore(selectSelectedImage);
  const canNavigateBack = useImageStore(selectCanNavigateBack);
  const canNavigateForward = useImageStore(selectCanNavigateForward);
  const navigateBack = useImageStore((s) => s.navigateBack);
  const navigateForward = useImageStore((s) => s.navigateForward);
  const showPreview = useImageStore((s) => s.showPreview);
  const setShowPreview = useImageStore((s) => s.setShowPreview);

  return (
    <Flex
      align="center"
      px="3"
      py="2"
      className="gap-0.5 border-b border-[var(--gray-6)]"
    >
      {/* Left: Navigation */}
      <ToolbarButton
        tooltip="Back"
        icon={<ArrowLeftIcon size={20} weight="bold" />}
        onClick={navigateBack}
        disabled={!canNavigateBack}
        ariaLabel="Back"
      />
      <ToolbarButton
        tooltip="Forward"
        icon={<ArrowRightIcon size={20} weight="bold" />}
        onClick={navigateForward}
        disabled={!canNavigateForward}
        ariaLabel="Forward"
      />

      <Box flexGrow="1" />

      {/* Center: Use actions */}
      {selectedImage && (
        <>
          <ToolbarButton
            tooltip="Use as source image"
            icon={<ImageIcon size={20} weight="fill" />}
            onClick={onUseAsSource}
            ariaLabel="Use as source image"
          />
          <ToolbarButton
            tooltip="Use as reference image"
            icon={<ImageSquareIcon size={20} weight="fill" />}
            onClick={onUseAsReference}
            ariaLabel="Use as reference image"
          />
          <DropdownMenu.Root>
            <DropdownMenu.Trigger>
              <Box>
                <ToolbarButton
                  tooltip="Use as ControlNet image"
                  icon={<SlidersHorizontalIcon size={20} weight="fill" />}
                  onClick={() => {}}
                  ariaLabel="Use as ControlNet image"
                />
              </Box>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {controlNets.map((_, index) => (
                <DropdownMenu.Item
                  key={index}
                  onSelect={() => onUseAsControlNet(index)}
                >
                  ControlNet {index + 1}
                </DropdownMenu.Item>
              ))}
              <DropdownMenu.Item onSelect={() => onUseAsControlNet(-1)}>
                New ControlNet
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
          <Box className="w-2 shrink-0" />
          <ToolbarButton
            tooltip="Use prompt from this image"
            icon={<QuotesIcon size={20} weight="fill" />}
            onClick={() =>
              params &&
              onUseParams(
                {
                  prompt: params.prompt,
                  negativePrompt: params.negativePrompt,
                },
                "prompt"
              )
            }
            disabled={!params}
            ariaLabel="Use prompt"
          />
          <ToolbarButton
            tooltip="Use seed from this image"
            icon={<AcornIcon size={20} weight="fill" />}
            onClick={() => params && onUseParams({ seed: params.seed }, "seed")}
            disabled={!params}
            ariaLabel="Use seed"
          />
          <ToolbarButton
            tooltip="Use aspect ratio from this image"
            icon={<RulerIcon size={20} weight="fill" />}
            onClick={onUseAspectRatio}
            disabled={!params}
            ariaLabel="Use aspect ratio"
          />
          <ToolbarButton
            tooltip="Use all parameters from this image"
            icon={<AsteriskIcon size={20} weight="bold" />}
            onClick={() => params && onUseParams(params, "all parameters")}
            disabled={!params}
            ariaLabel="Use all parameters"
          />
          <Box className="w-2 shrink-0" />
          <ToolbarButton
            tooltip="Face Detailer"
            icon={<UserFocusIcon size={20} weight="fill" />}
            onClick={onFaceDetailer}
            disabled={isGenerating}
            ariaLabel="Face Detailer"
          />
          <ToolbarButton
            tooltip="Upscale"
            icon={<ArrowsOutIcon size={20} weight="fill" />}
            onClick={onUpscale}
            disabled={isGenerating}
            ariaLabel="Upscale"
          />
        </>
      )}

      <Box flexGrow="1" />

      {/* Right: View toggles and delete */}
      <ToolbarButton
        tooltip="Show metadata"
        icon={<InfoIcon size={20} weight="fill" />}
        onClick={() => setShowMetadata(!showMetadata)}
        disabled={selectedImage == null || metadata == null}
        active={showMetadata}
        ariaLabel="Show metadata"
      />
      <ToolbarButton
        tooltip={showPreview ? "Hide preview" : "Show preview"}
        icon={<EyeIcon size={20} weight="fill" />}
        onClick={() => setShowPreview(!showPreview)}
        active={showPreview}
        ariaLabel="Toggle preview"
      />
      <Box className="w-2 shrink-0" />
      <ToolbarButton
        tooltip="Delete"
        icon={<TrashIcon size={20} weight="fill" />}
        onClick={onDelete}
        disabled={selectedImage == null}
        color="red"
        ariaLabel="Delete"
      />
    </Flex>
  );
}
