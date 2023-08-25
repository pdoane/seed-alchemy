import { FaShare } from "react-icons/fa";
import { useSnapshot } from "valtio";
import { IconButton } from "./components/Button";
import { toast } from "./components/Toast";
import { PromptGenResult } from "./schema";
import { stateSession, stateSettings, stateSystem } from "./store";
import { cx } from "./util/classNameUtil";

export const PromptResultsViewer = () => {
  const snapSession = useSnapshot(stateSession);

  function handleClick(result: PromptGenResult): void {
    console.log("clicked", result);
    result.used = true;
    stateSettings.generation.prompt.prompt = result.prompt;
    stateSystem.mode = "image";
    toast("Prompt set");
  }

  return (
    <div className="flex w-full h-full flex-shrink bg-black">
      <div className="flex flex-col space-y-3 p-3 overflow-y-scroll">
        {snapSession.promptGenResults.map((result, index) => (
          <div key={index} className="flex p-3 space-x-3 items-center bg-zinc-800 ">
            <IconButton icon={FaShare} onClick={() => handleClick(stateSession.promptGenResults[index])} />
            <label className={cx(result.used ? "text-gray-500" : "")}>{result.prompt}</label>
          </div>
        ))}
      </div>
    </div>
  );
};
