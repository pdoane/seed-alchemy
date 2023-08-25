import { useHotkeys } from "react-hotkeys-hook";
import { AiOutlineStop } from "react-icons/ai";
import { useSnapshot } from "valtio";
import { ControlNetParams } from "./ControlNetParams";
import { ImageFaceParams } from "./ImageFaceParams";
import { ImageGeneralParams } from "./ImageGeneralParams";
import { ImageHighResParams } from "./ImageHighResParams";
import { ImageInpaintParams } from "./ImageInpaintParams";
import { ImagePromptParams } from "./ImagePromptParams";
import { ImageRefinerParams } from "./ImageRefinerParams";
import { ImageSeedParams } from "./ImageSeedParams";
import { ImageSourceParams } from "./ImageSourceParams";
import { ImageUpscaleParams } from "./ImageUpscaleParams";
import { LoraParams } from "./LoraParams";
import { Button, IconButton } from "./components/Button";
import { useCancel, useGenerateImage, usePutSettings } from "./mutations";
import { useBaseModelType } from "./queries";
import { GenerationParamsState } from "./schema";

interface ImageGenerationParamsProps {
  state: GenerationParamsState;
  generatorId: string | null;
}

export const ImageGenerationParams = ({ state, generatorId }: ImageGenerationParamsProps) => {
  const snapGeneral = useSnapshot(state.general);
  const queryBaseModelType = useBaseModelType(snapGeneral.model);
  const postGenerateImage = useGenerateImage(state, generatorId);
  const postCancel = useCancel();
  const putSettings = usePutSettings();

  function onGenerate(): void {
    postGenerateImage.mutate();
    putSettings.mutate();
  }

  function onCancel(): void {
    postCancel.mutate();
  }

  useHotkeys(
    "mod+enter",
    onGenerate,
    {
      enabled: () => !postGenerateImage.isLoading,
      preventDefault: true,
      enableOnFormTags: ["input", "textarea", "select"],
    },
    [postGenerateImage.isLoading]
  );

  return (
    <>
      <div className="px-2 pt-2">
        <div className="flex space-x-2">
          <Button variant="primary" flexGrow={true} onClick={onGenerate} disabled={postGenerateImage.isLoading}>
            Generate
          </Button>
          <IconButton icon={AiOutlineStop} onClick={onCancel} disabled={!postGenerateImage.isLoading} />
        </div>
      </div>
      <div className="flex-grow px-2 pb-2 space-y-2 overflow-y-auto bg-zinc-900">
        <ImagePromptParams state={state.prompt} onGenerate={onGenerate} />
        <ImageGeneralParams state={state} />
        <ImageSeedParams state={state.seed} />
        <ImageSourceParams state={state.img2img} />
        <LoraParams state={state.lora} baseModelType={queryBaseModelType.data} />
        <ControlNetParams state={state.controlNet} baseModelType={queryBaseModelType.data} />
        <ImageHighResParams state={state.highRes} />
        <ImageRefinerParams state={state.refiner} baseModelType={queryBaseModelType.data} />
        <ImageUpscaleParams state={state.upscale} />
        <ImageFaceParams state={state.face} />
        <ImageInpaintParams state={state.inpaint} />
      </div>
    </>
  );
};
