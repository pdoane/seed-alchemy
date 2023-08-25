import { PromptGenerationPanel } from "./PromptGenerationPanel";
import { PromptResultsViewer } from "./PromptResultsViewer";
import { SessionProgress } from "./SessionProgress";
import { stateSettings } from "./store";

export const PromptMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden">
        <PromptGenerationPanel state={stateSettings.promptGen} stateView={stateSettings.promptGenView} />
        <PromptResultsViewer />
      </div>
    </div>
  );
};
