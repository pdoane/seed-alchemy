import { GenerationParamsState } from "./schema";
import { GenerationParams } from "./GenerationParams";

interface GenerationPanelProps {
  state: GenerationParamsState;
}

export const GenerationPanel = ({ state }: GenerationPanelProps) => {
  return (
    <div className="w-96 h-full flex-shrink-0 flex flex-col space-y-2 bg-zinc-900 ">
      <GenerationParams state={state} generatorId={null} />
    </div>
  );
};
