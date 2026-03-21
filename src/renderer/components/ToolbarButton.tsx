import { IconButton, Tooltip } from "@radix-ui/themes";
import type { ReactNode } from "react";

interface ToolbarButtonProps {
  tooltip: string;
  icon: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  color?: "gray" | "red" | "blue";
  ariaLabel: string;
}

export function ToolbarButton({
  tooltip,
  icon,
  onClick,
  disabled,
  active,
  color = "gray",
  ariaLabel,
}: ToolbarButtonProps) {
  const buttonColor = active ? "blue" : color;

  return (
    <Tooltip content={tooltip}>
      <IconButton
        size="2"
        variant="soft"
        color={buttonColor}
        disabled={disabled}
        onClick={onClick}
        aria-label={ariaLabel}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );
}
