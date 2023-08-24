import { useEffect, useState } from "react";
import { proxy, useSnapshot } from "valtio";
import { cx } from "../util/classNameUtil";

interface ToastItemState {
  id: number;
  active: boolean;
  type: "success" | "error" | "info";
  message: string;
}

const toastStore = proxy({
  toasts: [] as ToastItemState[],

  addToast: function (message: string, type: ToastItemState["type"]) {
    const id = Date.now();
    this.toasts.push({ id, active: true, message, type });
    setTimeout(() => this.removeToast(id), 3000);
  },

  removeToast: function (id: number | null) {
    const index = this.toasts.findIndex((toast) => toast.id === id);
    if (index !== -1) {
      const toast = this.toasts[index];
      if (toast.active) {
        this.toasts[index] = { ...toast, active: false };
        setTimeout(() => {
          for (let i = this.toasts.length - 1; i >= 0; i--) {
            const toast = this.toasts[i];
            if (toast.id === id) {
              this.toasts.splice(i, 1);
            }
          }
        }, 250);
      }
    }
  },
});

interface ToastAddProps {
  type?: ToastItemState["type"];
}

export function toast(message: string, props?: ToastAddProps) {
  const { type = "info" } = props || {};
  toastStore.addToast(message, type);
}

interface ToastItemProps {
  toast: ToastItemState;
}

const ToastItem = ({ toast }: ToastItemProps) => {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <div
      className={cx(
        "p-4 rounded",
        toast.type === "success" ? "bg-green-500" : toast.type === "error" ? "bg-red-500" : "bg-blue-500",
        isLoaded && toast.active ? "transition-opacity duration-250 opacity-100" : "opacity-0"
      )}
    >
      {toast.message}
      <button onClick={() => toastStore.removeToast(toast.id)} className="ml-4 text-white">
        X
      </button>
    </div>
  );
};

export const ToastContainer = () => {
  const snapshot = useSnapshot(toastStore);

  if (snapshot.toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 inset-x-0 px-4 flex flex-col space-y-1 items-center">
      {snapshot.toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
};
