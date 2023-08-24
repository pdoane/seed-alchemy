import { ChangeEvent } from "react";
import { cx } from "../util/classNameUtil";

interface SwitchProps {
  value: boolean;
  onChange?: (value: boolean) => void;
}

export const Switch = ({ value, onChange }: SwitchProps) => {
  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    if (onChange) onChange(event.target.checked);
  }

  return (
    <label className="relative inline-block w-10 align-middle select-none cursor-pointer transition duration-200 ease-in ">
      <input type="checkbox" className="sr-only" checked={value} onChange={handleChange} />
      <div className={cx("block w-10 h-5 rounded-full", value ? "bg-blue-600" : "bg-zinc-600")}></div>
      <div
        className={cx(
          "dot absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition",
          value ? "transform translate-x-5" : ""
        )}
      ></div>
    </label>
  );
};
