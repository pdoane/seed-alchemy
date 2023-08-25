import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { TextArea } from "./components/TextArea";
import { PromptParamsState } from "./schema";
import { KeyboardEvent } from "react";

interface ImagePromptParamsProps {
  state: PromptParamsState;
  onGenerate: () => void;
}

const PositivePromptParam = ({ state, onGenerate }: ImagePromptParamsProps) => {
  const snap = useSnapshot(state, { sync: true });

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && e.shiftKey === false) {
      e.preventDefault();
      onGenerate();
    }
  }

  return (
    <TextArea
      className="h-40"
      placeholder="Prompt"
      value={snap.prompt}
      onChange={(x) => (state.prompt = x)}
      onKeyDown={handleKeyDown}
    />
  );
};

const NegativePromptParam = ({ state, onGenerate }: ImagePromptParamsProps) => {
  const snap = useSnapshot(state, { sync: true });

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && e.shiftKey === false) {
      e.preventDefault();
      onGenerate();
    }
  }

  return (
    <TextArea
      placeholder="Negative Prompt"
      value={snap.negativePrompt}
      onChange={(x) => (state.negativePrompt = x)}
      onKeyDown={handleKeyDown}
    />
  );
};

export const ImagePromptParams = ({ state, onGenerate }: ImagePromptParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer label="Prompts" isOpen={snap.isOpen} onIsOpenChange={(x) => (state.isOpen = x)}>
      <PositivePromptParam state={state} onGenerate={onGenerate} />
      <NegativePromptParam state={state} onGenerate={onGenerate} />
    </CollapsibleContainer>
  );
};
