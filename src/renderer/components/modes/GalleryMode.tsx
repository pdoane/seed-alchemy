import { Box, Flex, Text } from "@radix-ui/themes";

export function GalleryMode() {
  return (
    <Flex
      height="100%"
      align="center"
      justify="center"
      className="bg-[var(--gray-1)]"
    >
      <Box className="text-center">
        <Text size="5" color="gray">
          Gallery Mode
        </Text>
        <Text as="p" size="2" color="gray" mt="2">
          Press Space for next image • Arrow keys to rate • Delete to remove
        </Text>
        <Text as="p" size="2" color="gray" mt="4">
          No images in gallery
        </Text>
      </Box>
    </Flex>
  );
}
