import { ChangeEvent } from "react";
import { cx } from "../util/classNameUtil";

interface TextInputProps {
  className?: string;
  label?: string;
  value: string;
  placeholder: string;
  onChange?: (value: string) => void;
}

export const TextInput = ({ label, value, placeholder, className, onChange }: TextInputProps) => {
  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    onChange?.(event.target.value);
  }

  return (
    <div className={cx("w-full flex flex-col space-y-1", className)}>
      {label && <label className="text-zinc-300">{label}</label>}
      <input
        id="textInput"
        type="text"
        className={cx(
          "p-1 bg-zinc-950",
          "focus:outline-none focus:ring-2 focus:ring-blue-500",
          "hover:ring-2 hover:ring-slate-500"
        )}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        autoComplete="off"
      />
    </div>
  );
};
