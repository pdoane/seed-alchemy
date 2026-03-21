import { Box, Flex } from "@radix-ui/themes";
import { useState, useEffect, useCallback } from "react";
import { ImageToolbar } from "./ImageToolbar";
import { ImageDisplay } from "./ImageDisplay";
import { ProgressBar } from "./ProgressBar";
import { ImageMetadataDisplay } from "./ImageMetadataDisplay";
import { useImageStore, selectSelectedImage } from "../store/imageStore";
import { api } from "../api/client";
import { toast } from "../store/toastStore";
import type { ImageMetadata, ImageParams } from "../../shared/types/image";
import { filterCompatibleLoras, remapControlNets } from "../../shared/utils";
import { applyAspectRatio } from "../lib/dimensions";

export function ImageViewport() {
  const selectedImage = useImageStore(selectSelectedImage);
  const removeImage = useImageStore((s) => s.removeImage);
  const setParams = useImageStore((s) => s.setParams);
  const setUi = useImageStore((s) => s.setUi);
  const isGenerating = useImageStore((s) => s.isGenerating);
  const generationStep = useImageStore((s) => s.generationStep);
  const generationMaxSteps = useImageStore((s) => s.generationMaxSteps);
  const generationNode = useImageStore((s) => s.generationNode);
  const availableCheckpoints = useImageStore((s) => s.availableCheckpoints);
  const availableLoras = useImageStore((s) => s.availableLoras);
  const currentLoras = useImageStore((s) => s.params.loras);
  const currentControlNets = useImageStore((s) => s.params.controlNets);
  const setSourceImage = useImageStore((s) => s.setSourceImage);
  const addReferenceImage = useImageStore((s) => s.addReferenceImage);
  const setControlNetImage = useImageStore((s) => s.setControlNetImage);
  const getCurrentArchitecture = useImageStore((s) => s.getCurrentArchitecture);
  const insertPromptFragment = useImageStore((s) => s.insertPromptFragment);
  const enhance = useImageStore((s) => s.enhance);

  const [showMetadata, setShowMetadata] = useState(false);
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null);

  const handleUseParams = useCallback(
    (params: Partial<ImageParams>, label: string) => {
      const arch = getCurrentArchitecture();
      const filteredParams = { ...params };

      // Filter incoming LoRAs for architecture compatibility
      if (params.loras) {
        const { compatible, incompatible } = filterCompatibleLoras(
          params.loras,
          arch,
          availableLoras
        );
        if (incompatible.length > 0) {
          toast.warning(
            `Skipped incompatible LoRAs: ${incompatible.join(", ")}`
          );
        }
        filteredParams.loras = compatible;
      }

      // Remap incoming ControlNets for architecture compatibility
      if (params.controlNets) {
        const { remapped, removedCount } = remapControlNets(
          params.controlNets,
          arch
        );
        if (removedCount > 0) {
          toast.warning(`Removed ${removedCount} incompatible ControlNet(s)`);
        }
        filteredParams.controlNets = remapped;
      }

      setParams(filteredParams);
      // Enable seedLocked when seed is applied
      if ("seed" in params) {
        setUi({ seedLocked: true });
      }
      toast.success(`Applied ${label}`);
    },
    [setParams, setUi, getCurrentArchitecture, availableLoras]
  );

  // Fetch metadata whenever image changes (needed for toolbar/context menu)
  useEffect(() => {
    if (selectedImage) {
      api
        .getImageMetadata(selectedImage)
        .then(setMetadata)
        .catch(() => setMetadata(null));
    } else {
      setMetadata(null);
    }
  }, [selectedImage]);

  const handleDelete = () => {
    if (selectedImage && confirm("Delete this image?")) {
      removeImage(selectedImage);
    }
  };

  const handleUseAsSource = useCallback(() => {
    if (selectedImage) {
      setSourceImage(selectedImage);
      toast.success("Set as source image");
    }
  }, [selectedImage, setSourceImage]);

  const handleUseAsReference = useCallback(() => {
    if (selectedImage) {
      addReferenceImage(selectedImage);
      toast.success("Added as reference image");
    }
  }, [selectedImage, addReferenceImage]);

  const handleUseSourceImage = useCallback(
    (filename: string) => {
      setSourceImage(filename);
      toast.success("Set as source image");
    },
    [setSourceImage]
  );

  const handleUseReferenceImage = useCallback(
    (filename: string) => {
      addReferenceImage(filename);
      toast.success("Added as reference image");
    },
    [addReferenceImage]
  );

  const handleUseControlNetImage = useCallback(
    (filename: string, index: number) => {
      setControlNetImage(index, filename);
      toast.success(`Set as ControlNet ${index + 1} image`);
    },
    [setControlNetImage]
  );

  const handleUseAsControlNet = useCallback(
    (index: number) => {
      if (selectedImage) {
        setControlNetImage(index, selectedImage);
        if (index === -1) {
          toast.success("Created new ControlNet");
        } else {
          toast.success(`Set as ControlNet ${index + 1} image`);
        }
      }
    },
    [selectedImage, setControlNetImage]
  );

  const handleUseAspectRatio = useCallback(() => {
    if (!metadata || metadata.operation.type !== "generate") return;
    const { width: imageWidth, height: imageHeight } =
      metadata.operation.params;
    const params = useImageStore.getState().params;
    const dims = applyAspectRatio(
      imageWidth,
      imageHeight,
      params.width,
      params.height,
      getCurrentArchitecture()
    );
    setParams(dims);
    toast.success("Applied aspect ratio");
  }, [metadata, getCurrentArchitecture, setParams]);

  const handleFaceDetailer = useCallback(async () => {
    if (!selectedImage) return;
    await enhance(selectedImage, "face_detailer");
  }, [selectedImage, enhance]);

  const handleUpscale = useCallback(async () => {
    if (!selectedImage) return;
    await enhance(selectedImage, "upscale");
  }, [selectedImage, enhance]);

  return (
    <Flex direction="column" height="100%" className="bg-[var(--gray-1)]">
      <ImageToolbar
        showMetadata={showMetadata}
        setShowMetadata={setShowMetadata}
        onDelete={handleDelete}
        metadata={metadata}
        onUseParams={handleUseParams}
        onUseAsSource={handleUseAsSource}
        onUseAsReference={handleUseAsReference}
        onUseAsControlNet={handleUseAsControlNet}
        onUseAspectRatio={handleUseAspectRatio}
        controlNets={currentControlNets || []}
        isGenerating={isGenerating}
        onFaceDetailer={handleFaceDetailer}
        onUpscale={handleUpscale}
      />
      <Box className="relative min-h-0 flex-1 overflow-hidden">
        <ImageDisplay />
        {isGenerating && (
          <ProgressBar
            step={generationStep}
            maxSteps={generationMaxSteps}
            node={generationNode}
          />
        )}
        {showMetadata && metadata && (
          <ImageMetadataDisplay
            metadata={metadata}
            checkpoints={availableCheckpoints}
            loras={availableLoras}
            currentLoras={currentLoras}
            currentControlNets={currentControlNets || []}
            onUseParams={handleUseParams}
            onUseAspectRatio={handleUseAspectRatio}
            onUseSourceImage={handleUseSourceImage}
            onUseReferenceImage={handleUseReferenceImage}
            onUseControlNetImage={handleUseControlNetImage}
            onInsertPromptFragment={insertPromptFragment}
          />
        )}
      </Box>
    </Flex>
  );
}
