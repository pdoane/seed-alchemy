import { Flex, IconButton, Tooltip } from "@radix-ui/themes";
import {
  CursorIcon,
  SelectionIcon,
  LassoIcon,
  MagicWandIcon,
  PaintBrushIcon,
  EraserIcon,
  SmileyStickerIcon,
  ScribbleIcon,
} from "@phosphor-icons/react";
import type { ToolType } from "../../../shared/types/document";

const toolIcons: Record<ToolType, React.ReactNode> = {
  select: <CursorIcon size={20} weight="fill" />,
  rectangle: <SelectionIcon size={20} weight="fill" />,
  lasso: <LassoIcon size={20} weight="fill" />,
  wand: <MagicWandIcon size={20} weight="fill" />,
  brush: <PaintBrushIcon size={20} weight="fill" />,
  eraser: <EraserIcon size={20} weight="fill" />,
  face: <SmileyStickerIcon size={20} weight="fill" />,
  scribble: <ScribbleIcon size={20} weight="fill" />,
};

const toolLabels: Record<ToolType, string> = {
  select: "Select (V)",
  rectangle: "Rectangle Select (M)",
  lasso: "Lasso (L)",
  wand: "Magic Wand (W)",
  brush: "Brush (B)",
  eraser: "Eraser (E)",
  face: "Face Select (F)",
  scribble: "Scribble (S)",
};

interface ToolPaletteProps {
  activeTool: ToolType;
  onToolChange: (tool: ToolType) => void;
}

export function ToolPalette({ activeTool, onToolChange }: ToolPaletteProps) {
  const toolOrder: ToolType[] = [
    "select",
    "rectangle",
    "lasso",
    "wand",
    "brush",
    "eraser",
    "face",
    "scribble",
  ];

  return (
    <Flex
      direction="column"
      align="center"
      gap="1"
      py="2"
      px="1"
      className="border-r border-[var(--gray-6)] bg-[var(--gray-2)]"
    >
      {toolOrder.map((toolId) => (
        <Tooltip key={toolId} content={toolLabels[toolId]} side="right">
          <IconButton
            size="2"
            variant={activeTool === toolId ? "solid" : "soft"}
            color={activeTool === toolId ? "violet" : "gray"}
            onClick={() => onToolChange(toolId)}
            aria-label={toolLabels[toolId]}
          >
            {toolIcons[toolId]}
          </IconButton>
        </Tooltip>
      ))}
    </Flex>
  );
}
