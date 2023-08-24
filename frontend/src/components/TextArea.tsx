import { ChangeEvent, KeyboardEvent } from "react";
import { cx } from "../util/classNameUtil";

type TextareaProps = {
  value: string;
  onChange?: (value: string) => void;
  placeholder: string;
  className?: string;
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
};

export const TextArea = ({ value, onChange, placeholder, className, onKeyDown }: TextareaProps) => {
  function handleChange(event: ChangeEvent<HTMLTextAreaElement>): void {
    onChange?.(event.target.value);
  }

  return (
    <textarea
      className={cx(
        "w-full block h-20 p-2",
        "bg-zinc-950",
        "focus:outline-none focus:ring-2 focus:ring-blue-500",
        "hover:ring-2 hover:ring-slate-500",
        "transition duration-250 ease-in-out",
        className
      )}
      placeholder={placeholder}
      value={value}
      onChange={handleChange}
      onKeyDown={onKeyDown}
    ></textarea>
  );
};
