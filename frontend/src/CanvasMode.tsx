import { CanvasLayerPanel } from "./CanvasLayerPanel";
import { CanvasScene } from "./CanvasScene";
import { CanvasSettingsPanel } from "./CanvasSettingsPanel";
import { CanvasToolbar } from "./CanvasToolbar";
import { SessionProgress } from "./SessionProgress";

export const CanvasMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden">
        <CanvasSettingsPanel />
        <CanvasToolbar />
        <CanvasScene />
        <CanvasLayerPanel />
      </div>
    </div>
  );
};
