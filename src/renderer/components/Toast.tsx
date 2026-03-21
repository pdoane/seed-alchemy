import { useEffect } from "react";
import {
  Cross2Icon,
  InfoCircledIcon,
  CheckCircledIcon,
  ExclamationTriangleIcon,
  CrossCircledIcon,
} from "@radix-ui/react-icons";
import { IconButton } from "@radix-ui/themes";
import { Toast as ToastType, useToastStore } from "../store/toastStore";

const DEFAULT_DURATION = 5000;

const typeStyles: Record<ToastType["type"], string> = {
  info: "bg-[var(--blue-3)] border-[var(--blue-6)]",
  success: "bg-[var(--green-3)] border-[var(--green-6)]",
  warning: "bg-[var(--yellow-3)] border-[var(--yellow-6)]",
  error: "bg-[var(--red-3)] border-[var(--red-6)]",
};

const typeIcons: Record<ToastType["type"], React.ReactNode> = {
  info: <InfoCircledIcon className="h-4 w-4 text-[var(--blue-11)]" />,
  success: <CheckCircledIcon className="h-4 w-4 text-[var(--green-11)]" />,
  warning: (
    <ExclamationTriangleIcon className="h-4 w-4 text-[var(--yellow-11)]" />
  ),
  error: <CrossCircledIcon className="h-4 w-4 text-[var(--red-11)]" />,
};

interface ToastProps {
  toast: ToastType;
}

export function Toast({ toast }: ToastProps) {
  const removeToast = useToastStore((state) => state.removeToast);
  const duration = toast.duration ?? DEFAULT_DURATION;

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        removeToast(toast.id);
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [toast.id, duration, removeToast]);

  return (
    <div
      role="alert"
      className={`flex items-center gap-3 rounded-md border px-4 py-3 shadow-lg ${typeStyles[toast.type]}`}
    >
      {typeIcons[toast.type]}
      <span className="flex-1 text-sm text-[var(--gray-12)]">
        {toast.message}
      </span>
      <IconButton
        size="1"
        variant="ghost"
        color="gray"
        onClick={() => removeToast(toast.id)}
        aria-label="Dismiss"
      >
        <Cross2Icon />
      </IconButton>
    </div>
  );
}
