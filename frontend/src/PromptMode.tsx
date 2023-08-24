import { SessionProgress } from "./SessionProgress";

export const PromptMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden items-center justify-center">Prompt Mode - Coming Soon</div>
    </div>
  );
};
