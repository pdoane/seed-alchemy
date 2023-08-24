import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { LoraModelParamsState, LoraParamsState } from "./schema";
import { LoraModelParams } from "./LoraModelParams";
import { useLoraModels } from "./queries";
import { Select, SelectItem } from "./components/Select";

interface LoraParamsProps {
  state: LoraParamsState;
  baseModelType: string | undefined;
}

export const LoraParams = ({ state, baseModelType }: LoraParamsProps) => {
  const snap = useSnapshot(state);
  const queryLoraModels = useLoraModels(baseModelType);

  function handleRemoveModel(index: number): void {
    state.entries.splice(index, 1);
  }

  function handleAddLora(value: string): void {
    state.entries.push(new LoraModelParamsState().load({ model: value }));
  }

  if (queryLoraModels.data?.length == 0) return <></>;

  return (
    <CollapsibleContainer
      label="LoRA"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <Select placeholder="Add LoRA" onChange={handleAddLora} status={queryLoraModels.status}>
        {queryLoraModels.data?.map((model) => (
          <SelectItem key={model} text={model} value={model} />
        ))}
      </Select>
      {snap.entries.map((entry, index) => (
        <LoraModelParams key={entry.model} state={state.entries[index]} onDelete={() => handleRemoveModel(index)} />
      ))}
    </CollapsibleContainer>
  );
};
