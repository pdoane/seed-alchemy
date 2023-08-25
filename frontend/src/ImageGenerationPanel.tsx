import { GenerationParamsState } from "./schema";
import { ImageGenerationParams } from "./ImageGenerationParams";

interface GenerationPanelProps {
  state: GenerationParamsState;
}

export const ImageGenerationPanel = ({ state }: GenerationPanelProps) => {
  return (
    <div className="w-96 h-full flex-shrink-0 flex flex-col space-y-2 bg-zinc-900 ">
      <ImageGenerationParams state={state} generatorId={null} />
    </div>
  );
};
