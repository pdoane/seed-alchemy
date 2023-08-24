import { useSnapshot } from "valtio";
import { Image } from "./components/Image";
import { QueryStatusLabel } from "./components/QueryStatus";
import { useMetadata } from "./queries";
import { CanvasElementState } from "./schema";
import { stateCanvas, stateSystem } from "./store";
import { cx } from "./util/classNameUtil";

interface CanvasImageElementProps {
  stateElement: CanvasElementState;
}

export const CanvasImageElement = ({ stateElement }: CanvasImageElementProps) => {
  const snapElement = useSnapshot(stateElement);
  const snapCanvas = useSnapshot(stateCanvas);
  const snapSystem = useSnapshot(stateSystem);
  const imagePath = snapElement.images[snapElement.imageIndex]?.path;
  const queryMetadata = useMetadata(imagePath);

  function handleMouseDown() {
    stateCanvas.selectedId = stateElement.id;
  }

  return (
    <div
      className={cx(
        "flex p-1 space-x-2 items-center justify-left hover:bg-blue-500",
        snapCanvas.selectedId == snapElement.id ? "bg-blue-600" : "bg-zinc-800"
      )}
      title={queryMetadata.data?.prompt}
      onMouseDown={handleMouseDown}
    >
      <div className="flex aspect-square w-16 h-16 items-center justify-center border border-zinc-500 bg-zinc-950">
        <Image src={`thumbnails/${snapSystem.user}/${imagePath}`} className="max-h-full max-w-full" />
      </div>
      {queryMetadata.data ? (
        <label className="truncate">{queryMetadata.data.prompt}</label>
      ) : (
        <QueryStatusLabel result={queryMetadata} />
      )}
    </div>
  );
};
