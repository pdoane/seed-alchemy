import { cx } from "../util/classNameUtil";
import { PointerEvent, useState } from "react";
import { SpinBox } from "./SpinBox";

interface SliderProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
  onChange: (value: number) => void;
}

export const Slider = ({ value, min = 0, max = 1, step = 0.01, decimals = 2, onChange }: SliderProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const updateValue = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const sliderValue = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
    const newValue = Math.round((min + (max - min) * sliderValue) / step) * step;

    onChange(newValue);
  };

  const handlePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    e.preventDefault();

    updateValue(e);
  };

  const handlePointerDown = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDragging(true);
    updateValue(e);
  };

  const handlePointerUp = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.releasePointerCapture(e.pointerId);
    setIsDragging(false);
  };

  const sliderPos = ((value - min) / (max - min)) * 100;

  return (
    <div className="w-full flex space-x-3 items-center">
      <div className="w-3/4 pl-2 pr-2">
        <div
          className="relative w-full h-4 cursor-pointer "
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onPointerMove={handlePointerMove}
        >
          <div className="absolute top-[5px] w-full h-1.5 bg-zinc-800"></div>
          <div className="absolute top-[5px] h-1.5 bg-blue-600" style={{ width: `${sliderPos}%` }}></div>
          <div
            className={cx(
              "absolute top-0 w-4 h-4 rounded-full cursor-pointer  hover:bg-blue-500",
              isDragging ? "bg-blue-500" : "bg-blue-600"
            )}
            style={{ left: `calc(${sliderPos}% - 0.5rem)` }}
          ></div>
          <input type="hidden" value={value} />
        </div>
      </div>
      <SpinBox
        className="w-1/4"
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        decimals={decimals}
      />
    </div>
  );
};
