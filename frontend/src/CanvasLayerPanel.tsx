import { stateCanvas } from "./store";
import { CanvasElement } from "./CanvasElement";
import { useSnapshot } from "valtio";
import { ToolbarButton } from "./components/Button";
import { FaPlus } from "react-icons/fa";
import { CanvasElementState, GenerationParamsState } from "./schema";

export const CanvasLayerPanel = () => {
  const snapCanvas = useSnapshot(stateCanvas);

  function handleNewGeneratorClick(): void {
    stateCanvas.elements.unshift(new CanvasElementState().load({ generation: new GenerationParamsState() }));
  }

  return (
    <div className="w-72 h-full flex-shrink-0 flex flex-col p-2 space-y-2 bg-zinc-900">
      <ToolbarButton icon={FaPlus} onClick={handleNewGeneratorClick} />
      <div>
        {snapCanvas.elements.map((element, index) => (
          <CanvasElement key={element.id} stateElement={stateCanvas.elements[index]} />
        ))}
      </div>
    </div>
  );
};
