import { Flex, Text } from "@radix-ui/themes";
import { useImageStore, selectSelectedImage } from "../store/imageStore";
import { api } from "../api/client";

export function ImageDisplay() {
  const selectedImage = useImageStore(selectSelectedImage);
  const isGenerating = useImageStore((s) => s.isGenerating);
  const previewUrl = useImageStore((s) => s.previewUrl);
  const showPreview = useImageStore((s) => s.showPreview);
  const generationWidth = useImageStore((s) => s.generationWidth);
  const generationHeight = useImageStore((s) => s.generationHeight);

  // Determine what to display - only one image at a time
  const showGenerationPreview = isGenerating && showPreview && previewUrl;
  const selectedImageUrl = selectedImage
    ? api.getImageUrl(selectedImage)
    : null;
  const displayUrl = showGenerationPreview ? previewUrl : selectedImageUrl;

  return (
    <Flex align="center" justify="center" p="4" className="h-full w-full">
      {displayUrl ? (
        <img
          src={displayUrl}
          alt="Generated image"
          className="max-h-full max-w-full select-none object-contain"
          style={
            showGenerationPreview
              ? {
                  width: generationWidth,
                  height: generationHeight,
                  imageRendering: "pixelated",
                }
              : undefined
          }
        />
      ) : (
        <Flex direction="column" align="center" gap="2">
          <Text size="6" color="gray">
            No Image Selected
          </Text>
          <Text size="2" color="gray">
            Select an image from the browser or generate a new one
          </Text>
        </Flex>
      )}
    </Flex>
  );
}
