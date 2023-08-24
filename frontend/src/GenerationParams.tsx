import { useHotkeys } from "react-hotkeys-hook";
import { AiOutlineStop } from "react-icons/ai";
import { ControlNetParams } from "./ControlNetParams";
import { FaceParams } from "./FaceParams";
import { GeneralParams } from "./GeneralParams";
import { Img2ImgParams } from "./Img2ImgParams";
import { PromptParams } from "./PromptParams";
import { SeedParams } from "./SeedParams";
import { UpscaleParams } from "./UpscaleParams";
import { Button, IconButton } from "./components/Button";
import { useCancel, useGenerateImage, usePutSettings } from "./mutations";
import { GenerationParamsState } from "./schema";
import { HighResParams } from "./HighResParams";
import { InpaintParams } from "./InpaintParams";
import { RefinerParams } from "./RefinerParams";
import { LoraParams } from "./LoraParams";
import { useBaseModelType } from "./queries";
import { useSnapshot } from "valtio";

interface GenerationParamsProps {
  state: GenerationParamsState;
  generatorId: string | null;
}

export const GenerationParams = ({ state, generatorId }: GenerationParamsProps) => {
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
        <PromptParams state={state.prompt} onGenerate={onGenerate} />
        <GeneralParams state={state} />
        <SeedParams state={state.seed} />
        <Img2ImgParams state={state.img2img} />
        <LoraParams state={state.lora} baseModelType={queryBaseModelType.data} />
        <ControlNetParams state={state.controlNet} baseModelType={queryBaseModelType.data} />
        <HighResParams state={state.highRes} />
        <RefinerParams state={state.refiner} baseModelType={queryBaseModelType.data} />
        <UpscaleParams state={state.upscale} />
        <FaceParams state={state.face} />
        <InpaintParams state={state.inpaint} />
      </div>
    </>
  );
};
