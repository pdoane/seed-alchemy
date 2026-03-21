import { Theme } from "@radix-ui/themes";
import { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { ImageMode } from "./components/modes/ImageMode";
import { DocumentMode } from "./components/modes/DocumentMode";
import { CanvasMode } from "./components/modes/CanvasMode";
import { GalleryMode } from "./components/modes/GalleryMode";
import { ModelsMode } from "./components/modes/ModelsMode";
import { ToastContainer } from "./components/ToastContainer";
import { ImageContextMenu } from "./components/ImageContextMenu";
import { useAppStore } from "./store/appStore";
import { initializeApp } from "./store/persistence";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

export function App() {
  const mode = useAppStore((state) => state.mode);

  useEffect(() => {
    return initializeApp();
  }, []);

  useKeyboardShortcuts();

  return (
    <Theme
      appearance="dark"
      accentColor="violet"
      grayColor="slate"
      radius="medium"
    >
      <div className="flex h-screen">
        <Sidebar />
        <main className="flex-1 overflow-hidden">
          {mode === "image" && <ImageMode />}
          {mode === "document" && <DocumentMode />}
          {mode === "canvas" && <CanvasMode />}
          {mode === "gallery" && <GalleryMode />}
          {mode === "models" && <ModelsMode />}
        </main>
      </div>
      <ToastContainer />
      <ImageContextMenu />
    </Theme>
  );
}
