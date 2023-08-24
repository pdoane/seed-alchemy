import { MouseEvent, useState } from "react";
import { FaWandSparkles } from "react-icons/fa6";
import { useSnapshot } from "valtio";
import { CanvasGeneratedImageMenu } from "./CanvasGeneratedImageMenu";
import { Image } from "./components/Image";
import { ContextMenu } from "./components/Menu";
import { CanvasElementState } from "./schema";
import { stateCanvas, stateSystem } from "./store";
import { cx } from "./util/classNameUtil";

interface CanvasGeneratorElementProps {
  stateElement: CanvasElementState;
}

export const CanvasGeneratorElement = ({ stateElement }: CanvasGeneratorElementProps) => {
  const snapElement = useSnapshot(stateElement);
  const snapCanvas = useSnapshot(stateCanvas);
  const snapGeneration = snapElement.generation!;
  const snapSystem = useSnapshot(stateSystem);
  const imagePath = snapElement.images[snapElement.imageIndex]?.path;
  const [contextMenuPoint, setContextMenuPoint] = useState<DOMPoint | null>(null);

  function handleMouseDown() {
    stateCanvas.selectedId = stateElement.id;
  }

  function handleImageClick(index: number): void {
    stateElement.imageIndex = index;
  }

  function handleContextMenu(event: MouseEvent<HTMLDivElement>, index: number): void {
    event.preventDefault();
    stateElement.imageIndex = index;
    setContextMenuPoint(new DOMPoint(event.clientX, event.clientY));
  }

  return (
    <>
      <div
        className={cx(
          "flex p-1 space-x-2 items-center justify-left hover:bg-blue-500",
          snapCanvas.selectedId == snapElement.id ? "bg-blue-600" : "bg-zinc-800"
        )}
        title={snapGeneration.prompt.prompt}
        onMouseDown={handleMouseDown}
      >
        <div
          className="flex aspect-square w-16 h-16 items-center justify-center border border-zinc-500 bg-zinc-800"
          style={{}}
        >
          <FaWandSparkles className="text-zinc-400 w-10 h-10" />
        </div>
        <label className="truncate">{snapGeneration.prompt.prompt}</label>
      </div>
      <div
        className="py-1 grid content-start gap-3"
        style={{
          gridTemplateColumns: "repeat(auto-fit, 80px)",
        }}
      >
        {snapElement.images.map((image, index) => (
          <div
            key={image.path}
            className={cx(
              "flex aspect-square items-center justify-center",
              snapElement.imageIndex == index ? "bg-slate-800" : "",
              "hover:outline hover:outline-zinc-500"
            )}
            onClick={() => handleImageClick(index)}
            onContextMenu={(e) => handleContextMenu(e, index)}
          >
            <Image src={`thumbnails/${snapSystem.user}/${image.path}`} className="max-h-full max-w-full" />
          </div>
        ))}
        {contextMenuPoint && (
          <ContextMenu point={contextMenuPoint} onClose={() => setContextMenuPoint(null)}>
            <CanvasGeneratedImageMenu imagePath={imagePath} />
          </ContextMenu>
        )}
      </div>
    </>
  );
};
