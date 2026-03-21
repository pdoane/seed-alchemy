import { Box, Flex, Heading, Text } from "@radix-ui/themes";
import { Stage, Layer, Rect, Text as KonvaText } from "react-konva";
import { useState, useEffect } from "react";

export function CanvasMode() {
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const updateDimensions = () => {
      const container = document.getElementById("canvas-container");
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: container.clientHeight,
        });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  return (
    <Flex height="100%">
      <Box id="canvas-container" flexGrow="1" className="bg-[var(--gray-1)]">
        <Stage width={dimensions.width} height={dimensions.height}>
          <Layer>
            <Rect
              x={0}
              y={0}
              width={dimensions.width}
              height={dimensions.height}
              fill="var(--gray-1)"
            />
            <KonvaText
              x={dimensions.width / 2 - 80}
              y={dimensions.height / 2 - 10}
              text="Infinite Canvas"
              fontSize={20}
              fill="#525252"
            />
          </Layer>
        </Stage>
      </Box>
      <Box
        width="320px"
        p="4"
        className="border-l border-[var(--gray-6)] bg-[var(--gray-2)]"
      >
        <Heading size="4" mb="4">
          Canvas Tools
        </Heading>
        <Text size="2" color="gray">
          Node tools and parameters will appear here
        </Text>
      </Box>
    </Flex>
  );
}
