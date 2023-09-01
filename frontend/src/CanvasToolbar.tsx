import { FaEraser, FaPaintBrush } from "react-icons/fa";
import { FaArrowPointer } from "react-icons/fa6";
import { useSnapshot } from "valtio";
import { canvasSceneRender } from "./CanvasScene";
import { Button, ModeButton } from "./components/Button";
import { RadioGroup } from "./components/RadioGroup";
import { stateCanvas } from "./store";
import { useUploadFile } from "./mutations";

export const CanvasToolbar = () => {
  const snapCanvas = useSnapshot(stateCanvas);
  const postUploadFile = useUploadFile(".canvas");

  function handleClearClick(): void {
    stateCanvas.strokes.length = 0;
  }

  function handleTestClick(): void {
    const element = stateCanvas.elements.find((x) => x.id == stateCanvas.selectedId);
    if (element === undefined || canvasSceneRender === null) return;

    const offscreenCanvas = canvasSceneRender.generateMask(element);
    offscreenCanvas.toBlob((blob) => {
      if (!blob) return;

      postUploadFile.mutate(blob);
    });
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
