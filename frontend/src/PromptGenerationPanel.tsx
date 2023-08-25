import { KeyboardEvent } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { useSnapshot } from "valtio";
import { Button } from "./components/Button";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Select, SelectItem } from "./components/Select";
import { Slider } from "./components/Slider";
import { SpinBox } from "./components/SpinBox";
import { TextArea } from "./components/TextArea";
import { useGeneratePrompt, usePutSettings } from "./mutations";
import { usePromptGenModels } from "./queries";
import { generateSeed } from "./random";
import { PromptGenRequest } from "./requests";
import { PromptGenViewState } from "./schema";

interface PromptGenerationPanelProps {
  state: PromptGenRequest;
  stateView: PromptGenViewState;
}

export const PromptGenerationPanel = ({ state, stateView }: PromptGenerationPanelProps) => {
  const snap = useSnapshot(state);
  const snapView = useSnapshot(stateView);
  const queryPromptGenModels = usePromptGenModels();
  const postGeneratePrompt = useGeneratePrompt(state, stateView);
  const putSettings = usePutSettings();

  function onGenerate(): void {
    postGeneratePrompt.mutate();
    putSettings.mutate();
  }

  function handleNewSeedClick(): void {
    state.seed = generateSeed();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && e.shiftKey === false) {
      e.preventDefault();
      onGenerate();
    }
  }

  useHotkeys(
    "mod+enter",
    onGenerate,
    {
      enabled: () => !postGeneratePrompt.isLoading,
      preventDefault: true,
      enableOnFormTags: ["input", "textarea", "select"],
    },
    [postGeneratePrompt.isLoading]
  );

  return (
    <div className="w-96 h-full flex-shrink-0 flex flex-col space-y-2 bg-zinc-900 ">
      <div className="px-2 pt-2">
        <div className="flex space-x-2">
          <Button variant="primary" flexGrow={true} onClick={onGenerate} disabled={postGeneratePrompt.isLoading}>
            Generate
          </Button>
        </div>
      </div>
      <div className="flex-grow px-2 pb-2 space-y-2 overflow-y-auto bg-zinc-900">
        <CollapsibleContainer
          label="Prompt"
          isOpen={snapView.promptIsOpen}
          onIsOpenChange={(x) => (stateView.promptIsOpen = x)}
        >
          <TextArea
            className="h-40"
            placeholder="Prompt"
            value={snap.prompt}
            onChange={(x) => (state.prompt = x)}
            onKeyDown={handleKeyDown}
          />
        </CollapsibleContainer>

        <CollapsibleContainer
          label="General"
          isOpen={snapView.generalIsOpen}
          onIsOpenChange={(x) => (stateView.generalIsOpen = x)}
        >
          <FormLabel label="Model">
            <Select value={snap.model} onChange={(x) => (state.model = x)} status={queryPromptGenModels.status}>
              {queryPromptGenModels.data?.map((model) => (
                <SelectItem key={model} text={model} value={model} />
              ))}
            </Select>
          </FormLabel>
          <FormLabel label="Prompt Count">
            <Slider value={snap.count} onChange={(x) => (state.count = x)} min={1} max={100} step={1} decimals={0} />
          </FormLabel>
          <FormLabel label="Temperature">
            <Slider value={snap.temperature} onChange={(x) => (state.temperature = x)} min={0.0} max={4.0} />
          </FormLabel>
          <FormLabel label="Top K">
            <Slider value={snap.top_k} onChange={(x) => (state.top_k = x)} min={1} max={50} step={1} decimals={0} />
          </FormLabel>
          <FormLabel label="Top P">
            <Slider value={snap.top_p} onChange={(x) => (state.top_p = x)} />
          </FormLabel>
          <FormLabel label="Beam Count">
            <Slider
              value={snap.beam_count}
              onChange={(x) => (state.beam_count = x)}
              min={1}
              max={8}
              step={1}
              decimals={0}
            />
          </FormLabel>
          <FormLabel label="Repetition Penalty">
            <Slider
              value={snap.repetition_penalty}
              onChange={(x) => (state.repetition_penalty = x)}
              min={0.0}
              max={4.0}
            />
          </FormLabel>
          <FormLabel label="Length Penalty">
            <Slider value={snap.length_penalty} onChange={(x) => (state.length_penalty = x)} min={-10.0} max={10.0} />
          </FormLabel>
          <FormLabel label="Min Length">
            <Slider
              value={snap.min_length}
              onChange={(x) => (state.min_length = x)}
              min={1}
              max={75}
              step={1}
              decimals={0}
            />
          </FormLabel>
          <FormLabel label="Max Length">
            <Slider
              value={snap.max_length}
              onChange={(x) => (state.max_length = x)}
              min={1}
              max={75}
              step={1}
              decimals={0}
            />
          </FormLabel>
        </CollapsibleContainer>
        <CollapsibleContainer
          label="Manual Seed"
          hasSwitch={true}
          isOpen={snapView.seedIsOpen}
          isEnabled={snapView.seedIsEnabled}
          onIsOpenChange={(x) => (stateView.seedIsOpen = x)}
          onIsEnabledChange={(x) => (stateView.seedIsEnabled = x)}
        >
          <FormLabel label="Seed">
            <div className="flex space-x-1">
              <SpinBox className="w-full" value={snap.seed} onChange={(x) => (state.seed = x)} min={1} />
              <Button onClick={handleNewSeedClick}>New</Button>
            </div>
          </FormLabel>
        </CollapsibleContainer>
      </div>
    </div>
  );
};
