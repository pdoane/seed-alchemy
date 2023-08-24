import { cx } from "../util/classNameUtil";

type ProgressBarProps = {
  value: number;
  max: number;
};

export const ProgressBar = ({ value, max }: ProgressBarProps) => {
  const indeterminate = value < 0;
  const percentage = indeterminate ? 100 : Math.round((value / max) * 100);

  return (
    <div className="w-full h-2 flex-shrink-0 bg-zinc-950">
      <div
        className={cx(
          "h-full bg-indigo-400 rounded",
          "transition-all duration-250",
          indeterminate ? "animate-pulse" : ""
        )}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
};
