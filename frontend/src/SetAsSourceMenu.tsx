import { useSnapshot } from "valtio";
import { Menu, MenuItem, MenuSeparator } from "./components/Menu";
import { toast } from "./components/Toast";
import { useMetadata } from "./queries";
import { updateAll } from "./requestUtil";
import { CanvasElementState, CanvasImage, ControlNetConditionParamsState, GenerationParamsState } from "./schema";
import { stateCanvas, stateSettings, stateSystem } from "./store";

interface SetAsSourceMenuProps {
  imagePath: string;
}

export const SetAsSourceMenu = ({ imagePath }: SetAsSourceMenuProps) => {
  const stateGeneration = stateSettings.generation;
  const snapGeneration = useSnapshot(stateGeneration);
  const queryMetadata = useMetadata(imagePath);

  function handleImageToImageClick(): void {
    stateGeneration.img2img.source = imagePath;
    stateGeneration.img2img.isEnabled = true;
    toast("Image set as Source");
  }

  function handleControlNetClick(index: number): void {
    stateGeneration.controlNet.conditions[index].source = imagePath;
    stateGeneration.controlNet.isEnabled = true;
    stateGeneration.controlNet.activeTab = index;
    toast("Image set for ControlNet");
  }

  function handleControlNetNewClick(): void {
    stateGeneration.controlNet.conditions.push(new ControlNetConditionParamsState().load({ source: imagePath }));
    stateGeneration.controlNet.isEnabled = true;
    stateGeneration.controlNet.activeTab = stateGeneration.controlNet.conditions.length - 1;
    toast("Image set for ControlNet");
  }

  function handleCanvasClick(): void {
    if (queryMetadata.data) {
      const metadata = queryMetadata.data;
      const generation = new GenerationParamsState();
      updateAll(generation, metadata);

      stateCanvas.elements.unshift(
        new CanvasElementState().load({
          width: metadata.width,
          height: metadata.height,
          images: [new CanvasImage().load({ path: imagePath })],
        })
      );
      stateCanvas.elements.unshift(
        new CanvasElementState().load({
          width: metadata.width,
          height: metadata.height,
          generation,
        })
      );
      stateSystem.mode = "canvas";
      toast("Image added to Canvas");
    }
  }

  return (
    <Menu>
      <MenuItem text="Image to Image" onClick={handleImageToImageClick} />
      {snapGeneration.controlNet.conditions.map((condition, index) => (
        <MenuItem key={condition.id} text={`Control Net ${index + 1}`} onClick={() => handleControlNetClick(index)} />
      ))}
      <MenuItem text="Control Net (New Condition)" onClick={handleControlNetNewClick} />
      <MenuSeparator />
      <MenuItem text="Canvas" disabled={!queryMetadata.isSuccess} onClick={handleCanvasClick} />
      <MenuItem text="Interrogate" disabled={true} />
    </Menu>
  );
};
