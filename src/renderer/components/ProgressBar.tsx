import { Box, Flex, Text } from "@radix-ui/themes";

interface ProgressBarProps {
  step: number;
  maxSteps: number;
  node: string | null;
}

export function ProgressBar({ step, maxSteps, node }: ProgressBarProps) {
  const progress = maxSteps > 0 ? (step / maxSteps) * 100 : 0;

  return (
    <Box
      className="absolute left-0 right-0 top-0 p-2"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.6)" }}
    >
      <Flex align="center" gap="2">
        {node && (
          <Text size="1" className="text-white">
            {node}
          </Text>
        )}
        <Box
          className="h-1 flex-1 overflow-hidden rounded"
          style={{ backgroundColor: "var(--gray-7)" }}
        >
          <Box
            className="h-full transition-all"
            style={{
              backgroundColor: "var(--blue-9)",
              width: `${progress}%`,
            }}
          />
        </Box>
        <Text size="1" className="text-white">
          {step}/{maxSteps}
        </Text>
      </Flex>
    </Box>
  );
}
