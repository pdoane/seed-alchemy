import { DropdownMenu, Flex } from "@radix-ui/themes";
import {
  XIcon,
  ImageIcon,
  ImageSquareIcon,
  SlidersHorizontalIcon,
  QuotesIcon,
  AcornIcon,
  RulerIcon,
  AsteriskIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useImageStore } from "../store/imageStore";
import { toast } from "../store/toastStore";
import type { ImageMetadata, ImageParams } from "../../shared/types/image";
import { applyAspectRatio } from "../lib/dimensions";

// Helper component for menu items with icons
function MenuItem({
  icon,
  children,
  ...props
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
} & React.ComponentProps<typeof DropdownMenu.Item>) {
  return (
    <DropdownMenu.Item {...props}>
      <Flex align="center" gap="2">
        {icon}
        {children}
      </Flex>
    </DropdownMenu.Item>
  );
}

export function ImageContextMenu() {
  // Subscribe to context menu state
  const contextMenu = useImageStore((s) => s.contextMenu);
  const closeContextMenu = useImageStore((s) => s.closeContextMenu);

  // Subscribe to action functions (stable references, no re-renders)
  const setParams = useImageStore((s) => s.setParams);
  const removeImage = useImageStore((s) => s.removeImage);
  const addReferenceImage = useImageStore((s) => s.addReferenceImage);
  const setSourceImage = useImageStore((s) => s.setSourceImage);
  const setControlNetImage = useImageStore((s) => s.setControlNetImage);
  const reorderReferenceImages = useImageStore((s) => s.reorderReferenceImages);
  const getCurrentArchitecture = useImageStore((s) => s.getCurrentArchitecture);

  const [metadata, setMetadata] = useState<ImageMetadata | null>(null);

  // Extract params if this is a generated image
  const imgParams =
    metadata?.operation.type === "generate"
      ? metadata.operation.params
      : undefined;

  // Fetch metadata when context menu opens
  useEffect(() => {
    if (contextMenu?.filename) {
      api
        .getImageMetadata(contextMenu.filename)
        .then(setMetadata)
        .catch(() => setMetadata(null));
    } else {
      setMetadata(null);
    }
  }, [contextMenu?.filename]);

  if (!contextMenu) return null;

  const {
    filename,
    context,
    position,
    referenceIndex,
    referenceCount,
    onRemove,
  } = contextMenu;

  // Get values lazily (they change frequently, don't need reactivity here)
  const architecture = getCurrentArchitecture();
  const controlNets = useImageStore.getState().params.controlNets;

  const handleUseParams = (params: Partial<ImageParams>, label: string) => {
    setParams(params);
    toast.success(`Applied ${label}`);
    closeContextMenu();
  };

  const handleDelete = () => {
    if (confirm("Delete this image?")) {
      removeImage(filename);
    }
    closeContextMenu();
  };

  const handleUseAsReference = () => {
    addReferenceImage(filename);
    toast.success("Added as reference image");
    closeContextMenu();
  };

  const handleUseAsSource = () => {
    setSourceImage(filename);
    toast.success("Set as source image");
    closeContextMenu();
  };

  const handleUseAsControlNet = (index: number) => {
    setControlNetImage(index, filename);
    if (index === -1) {
      toast.success("Created new ControlNet");
    } else {
      toast.success(`Set as ControlNet ${index + 1} image`);
    }
    closeContextMenu();
  };

  const handleUseAspectRatio = () => {
    if (!imgParams) return;
    const { width: imageWidth, height: imageHeight } = imgParams;
    const { params } = useImageStore.getState();
    const dims = applyAspectRatio(
      imageWidth,
      imageHeight,
      params.width,
      params.height,
      getCurrentArchitecture()
    );
    setParams(dims);
    toast.success("Applied aspect ratio");
    closeContextMenu();
  };

  const handleMoveUp = () => {
    if (referenceIndex !== undefined && referenceIndex > 0) {
      reorderReferenceImages(referenceIndex, referenceIndex - 1);
    }
    closeContextMenu();
  };

  const handleMoveDown = () => {
    if (
      referenceIndex !== undefined &&
      referenceCount !== undefined &&
      referenceIndex < referenceCount - 1
    ) {
      reorderReferenceImages(referenceIndex, referenceIndex + 1);
    }
    closeContextMenu();
  };

  const handleRemove = () => {
    onRemove?.();
    closeContextMenu();
  };
  const canMoveUp = referenceIndex !== undefined && referenceIndex > 0;
  const canMoveDown =
    referenceIndex !== undefined &&
    referenceCount !== undefined &&
    referenceIndex < referenceCount - 1;

  return (
    <div
      className="fixed inset-0 z-50"
      onClick={closeContextMenu}
      onContextMenu={(e) => {
        e.preventDefault();
        closeContextMenu();
      }}
    >
      <div
        style={{
          position: "absolute",
          left: position.x,
          top: position.y,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <DropdownMenu.Root
          open
          onOpenChange={(open) => !open && closeContextMenu()}
        >
          <DropdownMenu.Trigger>
            <span />
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="start">
            {/* Use as Source/Reference */}
            <MenuItem
              icon={<ImageIcon size={14} />}
              disabled={context === "source"}
              onClick={handleUseAsSource}
            >
              Use as Source Image
            </MenuItem>
            <MenuItem
              icon={<ImageSquareIcon size={14} />}
              disabled={context === "reference"}
              onClick={handleUseAsReference}
            >
              Use as Reference Image
            </MenuItem>
            {architecture === "sdxl" && (
              <DropdownMenu.Sub>
                <DropdownMenu.SubTrigger disabled={context === "controlnet"}>
                  <Flex align="center" gap="2">
                    <SlidersHorizontalIcon size={14} />
                    Use as ControlNet Image
                  </Flex>
                </DropdownMenu.SubTrigger>
                <DropdownMenu.SubContent>
                  {(controlNets || []).map((_, index) => (
                    <DropdownMenu.Item
                      key={index}
                      onClick={() => handleUseAsControlNet(index)}
                    >
                      ControlNet {index + 1}
                    </DropdownMenu.Item>
                  ))}
                  <DropdownMenu.Item onClick={() => handleUseAsControlNet(-1)}>
                    New ControlNet
                  </DropdownMenu.Item>
                </DropdownMenu.SubContent>
              </DropdownMenu.Sub>
            )}

            <DropdownMenu.Separator />

            {/* Use parameters from image */}
            <MenuItem
              icon={<QuotesIcon size={14} />}
              disabled={!imgParams}
              onClick={() =>
                imgParams &&
                handleUseParams(
                  {
                    prompt: imgParams.prompt,
                    negativePrompt: imgParams.negativePrompt,
                  },
                  "prompt"
                )
              }
            >
              Use Prompt
            </MenuItem>
            <MenuItem
              icon={<AcornIcon size={14} />}
              disabled={!imgParams}
              onClick={() =>
                imgParams && handleUseParams({ seed: imgParams.seed }, "seed")
              }
            >
              Use Seed
            </MenuItem>
            <MenuItem
              icon={<RulerIcon size={14} />}
              disabled={!imgParams}
              onClick={handleUseAspectRatio}
            >
              Use Aspect Ratio
            </MenuItem>
            <MenuItem
              icon={<AsteriskIcon size={14} weight="bold" />}
              disabled={!imgParams}
              onClick={() =>
                imgParams && handleUseParams(imgParams, "all parameters")
              }
            >
              Use All
            </MenuItem>

            {/* Move Up/Down for reference images */}
            {context === "reference" && (
              <>
                <DropdownMenu.Separator />
                <MenuItem
                  icon={<ArrowUpIcon size={14} />}
                  disabled={!canMoveUp}
                  onClick={handleMoveUp}
                >
                  Move Up
                </MenuItem>
                <MenuItem
                  icon={<ArrowDownIcon size={14} />}
                  disabled={!canMoveDown}
                  onClick={handleMoveDown}
                >
                  Move Down
                </MenuItem>
              </>
            )}

            <DropdownMenu.Separator />

            {/* Delete/Remove */}
            {context === "gallery" ? (
              <MenuItem
                icon={<TrashIcon size={14} />}
                color="red"
                onClick={handleDelete}
              >
                Delete
              </MenuItem>
            ) : (
              <MenuItem
                icon={<XIcon size={14} />}
                color="red"
                onClick={handleRemove}
              >
                Remove
              </MenuItem>
            )}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
    </div>
  );
}
