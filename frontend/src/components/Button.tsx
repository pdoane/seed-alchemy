import { ChangeEvent, MouseEvent, ReactNode, forwardRef, useContext, useRef, useState } from "react";
import { IconType } from "react-icons";
import { cx } from "../util/classNameUtil";
import { ContextMenu } from "./Menu";
import { RadioGroupContext } from "./RadioGroup";

type ButtonProps = {
  variant?: "primary" | "solid";
  children?: ReactNode;
  disabled?: boolean;
  flexGrow?: boolean;
  onClick?: () => void;
};

type IconButtonProps = {
  icon: IconType;
  disabled?: boolean;
  onClick?: () => void;
};

type ModeButtonProps = {
  icon: IconType;
  value: string;
};

type ToolbarButtonProps = {
  icon: IconType;
  disabled?: boolean;
  onClick?: () => void;
  menu?: ReactNode;
  placement?: string;
};

type ToolbarCheckButtonProps = {
  icon: IconType;
  value: boolean;
  onChange?: (value: boolean) => void;
};

interface CheckboxProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "solid", children, disabled = false, flexGrow = false, onClick }: ButtonProps, ref) => {
    function handleClick(event: MouseEvent<HTMLButtonElement>): void {
      onClick?.();
      event.stopPropagation();
    }

    return (
      <button
        ref={ref}
        className={cx(
          flexGrow ? "flex-grow" : "",
          "py-1 px-2",
          disabled
            ? "cursor-not-allowed bg-gray-700"
            : variant == "primary"
            ? "bg-blue-700 hover:bg-blue-600"
            : "bg-zinc-950 hover:bg-slate-800",
          "text-sm",
          "text-white",
          "transition duration-250 ease-in-out"
        )}
        disabled={disabled}
        onClick={handleClick}
      >
        {children}
      </button>
    );
  }
);

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon, disabled = false, onClick }: IconButtonProps, ref) => {
    const Icon = icon;
    function handleClick(event: MouseEvent<HTMLButtonElement>): void {
      onClick?.();
      event.stopPropagation();
    }

    return (
      <button
        ref={ref}
        className={cx(
          "py-1 px-2",
          disabled ? "cursor-not-allowed bg-gray-700" : "bg-zinc-950 hover:bg-slate-800",
          "text-base",
          "text-white",
          "transition duration-250 ease-in-out"
        )}
        disabled={disabled}
        onClick={handleClick}
      >
        <Icon />
      </button>
    );
  }
);

export const ModeButton = ({ icon, value }: ModeButtonProps) => {
  const Icon = icon;
  const group = useContext(RadioGroupContext);

  const isChecked = group.value == value;

  return (
    <button
      className={cx(
        "w-10 h-10 flex items-center justify-center text-2xl text-white focus:outline-none",
        isChecked ? " bg-blue-600" : "hover:bg-zinc-600",
        "text-white",
        "transition duration-250 ease-in-out"
      )}
      onClick={() => group.onChange?.(value)}
      tabIndex={-1}
    >
      <Icon />
    </button>
  );
};

export const ToolbarButton = ({ icon, disabled = false, onClick, menu, placement }: ToolbarButtonProps) => {
  const Icon = icon;
  const [contextMenuPoint, setContextMenuPoint] = useState<DOMPoint | null>(null);
  const ref = useRef<HTMLButtonElement>(null);

  function handleClick(event: MouseEvent<HTMLButtonElement>): void {
    onClick?.();
    event.stopPropagation();
    if (menu) {
      const rect = event.currentTarget.getBoundingClientRect();
      if (placement === "right") {
        setContextMenuPoint(new DOMPoint(rect.right, rect.top));
      } else {
        setContextMenuPoint(new DOMPoint(rect.left, rect.bottom));
      }
    }
  }

  return (
    <div>
      <button
        ref={ref}
        className={cx(
          "w-10 h-10 flex items-center justify-center text-lg",
          disabled ? "text-gray-500 cursor-not-allowed" : "text-white hover:bg-zinc-600",
          contextMenuPoint ? "bg-slate-600" : "",
          "transition duration-250 ease-in-out"
        )}
        disabled={disabled}
        onClick={handleClick}
        tabIndex={-1}
      >
        <Icon />
      </button>
      {contextMenuPoint && (
        <ContextMenu point={contextMenuPoint} onClose={() => setContextMenuPoint(null)}>
          {menu}
        </ContextMenu>
      )}
    </div>
  );
};

export const ToolbarCheckButton = ({ icon, value, onChange }: ToolbarCheckButtonProps) => {
  const Icon = icon;

  function handleClick(event: MouseEvent<HTMLButtonElement>): void {
    onChange?.(!value);
    event.stopPropagation();
  }

  return (
    <button
      className={cx(
        "w-10 h-10 flex items-center justify-center text-lg text-white focus:outline-none",
        value ? " bg-blue-600 hover:bg-blue-500" : "hover:bg-zinc-600",
        "text-white",
        "transition duration-250 ease-in-out"
      )}
      onClick={handleClick}
      tabIndex={-1}
    >
      <Icon />
    </button>
  );
};

export const Checkbox = ({ label, value, onChange }: CheckboxProps) => {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.checked);
  };

  return (
    <label className="flex items-center space-x-2 select-none">
      <input type="checkbox" checked={value} onChange={handleChange} className="form-checkbox h-5 w-5 text-blue-600" />
      <span>{label}</span>
    </label>
  );
};
