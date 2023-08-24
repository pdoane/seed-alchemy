import { useSnapshot } from "valtio";
import { AppToolbar } from "./AppToolbar";
import { CanvasMode } from "./CanvasMode";
import { DeleteDialog } from "./DeleteDialog";
import { GalleryMode } from "./GalleryMode";
import { ImageHistory } from "./ImageHistory";
import { ImageMode } from "./ImageMode";
import { InterrogateMode } from "./InterrogateMode";
import { PromptMode } from "./PromptMode";
import { SettingsDialog } from "./SettingsDialog";
import { ToastContainer } from "./components/Toast";
import { stateSession, stateSystem } from "./store";
import { WebSocketComponent } from "./websocket";

export const App = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSystem = useSnapshot(stateSystem);

  return (
    <div className="flex w-screen h-screen">
      <WebSocketComponent />
      <ImageHistory />
      <AppToolbar />
      {snapSystem.mode === "image" && <ImageMode />}
      {snapSystem.mode === "canvas" && <CanvasMode />}
      {snapSystem.mode === "gallery" && <GalleryMode />}
      {snapSystem.mode === "prompt" && <PromptMode />}
      {snapSystem.mode === "interrogate" && <InterrogateMode />}
      <ToastContainer />
      {snapSession.dialog == "settings" && <SettingsDialog />}
      {snapSession.dialog == "delete" && <DeleteDialog />}
      <div id="context-menu-root"></div>
    </div>
  );
};
