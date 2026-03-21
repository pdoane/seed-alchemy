import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { ImageViewport } from "../ImageViewport";
import { ParameterPanel } from "../ParameterPanel";
import { ImageBrowser } from "../ImageBrowser";

export function ImageMode() {
  return (
    <PanelGroup
      direction="horizontal"
      className="h-full"
      autoSaveId="image-mode"
    >
      <Panel defaultSize={20} minSize={20} maxSize={40}>
        <ParameterPanel />
      </Panel>
      <PanelResizeHandle className="w-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />
      <Panel minSize={30}>
        <ImageViewport />
      </Panel>
      <PanelResizeHandle className="w-1 bg-[var(--gray-6)] transition-colors hover:bg-[var(--accent-8)]" />
      <Panel defaultSize={15} minSize={10} maxSize={30}>
        <ImageBrowser />
      </Panel>
    </PanelGroup>
  );
}
