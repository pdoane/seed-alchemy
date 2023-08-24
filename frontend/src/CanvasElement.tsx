import { useSnapshot } from "valtio";
import { CanvasGeneratorElement } from "./CanvasGeneratorElement";
import { CanvasImageElement } from "./CanvasImageElement";
import { CanvasElementState } from "./schema";

interface CanvasElementProps {
  stateElement: CanvasElementState;
}

export const CanvasElement = ({ stateElement }: CanvasElementProps) => {
  const snapElement = useSnapshot(stateElement);

  return (
    <div className="">
      {snapElement.generation ? (
        <CanvasGeneratorElement stateElement={stateElement} />
      ) : (
        <CanvasImageElement stateElement={stateElement} />
      )}
    </div>
  );
};
