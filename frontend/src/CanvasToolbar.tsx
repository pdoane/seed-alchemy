import { FaEraser, FaPaintBrush } from "react-icons/fa";
import { FaArrowPointer } from "react-icons/fa6";
import { useSnapshot } from "valtio";
import { canvasSceneRender } from "./CanvasScene";
import { Button, ModeButton } from "./components/Button";
import { RadioGroup } from "./components/RadioGroup";
import { stateCanvas } from "./store";

export const CanvasToolbar = () => {
  const snapCanvas = useSnapshot(stateCanvas);

  function handleClearClick(): void {
    stateCanvas.strokes.length = 0;
  }

  function handleTestClick(): void {
    canvasSceneRender?.generateMask();
  }

  return (
    <div className="h-full flex flex-col justify-between bg-zinc-950">
      <div>
        <RadioGroup value={snapCanvas.tool} onChange={(x) => (stateCanvas.tool = x)}>
          <ModeButton value="select" icon={FaArrowPointer} />
          <ModeButton value="brush" icon={FaPaintBrush} />
          <ModeButton value="eraser" icon={FaEraser} />
        </RadioGroup>
        <Button onClick={handleClearClick}>Clear</Button>
        <Button onClick={handleTestClick}>Test</Button>
      </div>
    </div>
  );
};
