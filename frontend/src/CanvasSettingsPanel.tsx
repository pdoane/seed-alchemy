import { useEffect } from "react";
import { GenerationParams } from "./GenerationParams";
import { MetadataViewer } from "./MetadataViewer";
import { CanvasElementState } from "./schema";
import { stateCanvas } from "./store";
import { subscribe, useSnapshot } from "valtio";

interface CanvasElementSettingsProps {
  stateElement: CanvasElementState;
}

export const CanvasGenerationSettings = ({ stateElement }: CanvasElementSettingsProps) => {
  const stateGeneration = stateElement.generation!;

  useEffect(() => {
    const unsubscribe = subscribe(stateGeneration.general, () => {
      stateElement.width = stateGeneration.general.width;
      stateElement.height = stateGeneration.general.height;
    });

    return () => {
      unsubscribe();
    };
  }, [stateGeneration]);

  return <GenerationParams state={stateGeneration} generatorId={stateElement.id} />;
};

export const CanvasImageSettings = ({ stateElement }: CanvasElementSettingsProps) => {
  const snapElement = useSnapshot(stateElement);

  return (
    <div className="p-2 overflow-y-scroll">
      <MetadataViewer path={snapElement.images[snapElement.imageIndex].path} />
    </div>
  );
};

export const CanvasElementSettings = ({ stateElement }: CanvasElementSettingsProps) => {
  const snapElement = useSnapshot(stateElement);

  return snapElement.generation ? (
    <CanvasGenerationSettings stateElement={stateElement} />
  ) : (
    <CanvasImageSettings stateElement={stateElement} />
  );
};

export const CanvasSettingsPanel = () => {
  const snapCanvas = useSnapshot(stateCanvas);
  const index = snapCanvas.elements.findIndex((element) => element.id == snapCanvas.selectedId);

  return (
    <div className="w-96 h-full flex-shrink-0 flex flex-col space-y-2 bg-zinc-900 ">
      {index != -1 && <CanvasElementSettings stateElement={stateCanvas.elements[index]} />}
    </div>
  );
};
