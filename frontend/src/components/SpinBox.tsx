import { ChangeEvent, KeyboardEvent, useCallback, useState } from "react";
import { AiFillCaretDown, AiFillCaretUp } from "react-icons/ai";
import { useInterval } from "../hooks/useInterval";
import { cx } from "../util/classNameUtil";

interface SpinBoxProps {
  className?: string;
  value: number;
  onChange?: (value: number) => void;
  min: number;
  max?: number;
  step?: number;
  decimals?: number;
}

export const SpinBox = ({
  className,
  value,
  onChange,
  min,
  max = Number.MAX_SAFE_INTEGER,
  step = 1,
  decimals = 0,
}: SpinBoxProps) => {
  const [isFocused, setIsFocused] = useState(false);
  const [delay, setDelay] = useState<null | number>(null);
  const [action, setAction] = useState<"increment" | "decrement" | null>(null);

  const increment = () => {
    onChange?.(Math.min(value + step, max));
  };

  const decrement = () => {
    onChange?.(Math.max(value - step, min));
  };

  const handleFocus = () => setIsFocused(true);
  const handleBlur = () => setIsFocused(false);

  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    onChange?.(Number(event.target.value));
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    switch (event.key) {
      case "ArrowUp":
        event.preventDefault();
        increment();
        break;
      case "ArrowDown":
        event.preventDefault();
        decrement();
        break;
      default:
        break;
    }
  }

  useInterval(() => {
    if (action === "increment") {
      increment();
    } else if (action === "decrement") {
      decrement();
    }
  }, delay);

  const startIncrement = useCallback(() => {
    setAction("increment");
    setDelay(200);
  }, []);

  const startDecrement = useCallback(() => {
    setAction("decrement");
    setDelay(200);
  }, []);

  const stopInterval = useCallback(() => {
    setDelay(null);
  }, []);

  return (
    <div
      className={cx(
        "flex bg-zinc-950",
        isFocused ? "ring-2 ring-blue-500" : "hover:ring-2 hover:ring-slate-500",
        className
      )}
      onFocus={handleFocus}
      onBlur={handleBlur}
    >
      <input
        type="text"
        value={value.toFixed(decimals)}
        onKeyDown={handleKeyDown}
        onChange={handleChange}
        className="w-full p-1 outline-none bg-zinc-950 text-center"
      />
      <div className="flex flex-col pr-0.5">
        <button
          tabIndex={-1}
          className={value >= max ? "cursor-not-allowed text-gray-400" : ""}
          onMouseDown={startIncrement}
          onMouseUp={stopInterval}
          onMouseLeave={stopInterval}
          onClick={increment}
        >
          <AiFillCaretUp />
        </button>
        <button
          tabIndex={-1}
          className={value <= min ? "cursor-not-allowed text-gray-400" : ""}
          onMouseDown={startDecrement}
          onMouseUp={stopInterval}
          onMouseLeave={stopInterval}
          onClick={decrement}
        >
          <AiFillCaretDown />
        </button>
      </div>
    </div>
  );
};
