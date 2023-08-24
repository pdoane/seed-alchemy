import { ReactNode } from "react";
import { cx } from "../util/classNameUtil";
import { useHotkeys } from "react-hotkeys-hook";

interface ModalProps {
  onClose: () => void;
  children: ReactNode;
  className?: string;
}

export const Modal = ({ onClose, children, className }: ModalProps) => {
  useHotkeys("esc", onClose, []);

  return (
    <div
      className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-50"
      onClick={onClose}
    >
      <div className={cx("p-2 space-y-2 rounded bg-zinc-800", className)} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
};
