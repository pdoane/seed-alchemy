import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Select, SelectItem } from "./components/Select";
import { Slider } from "./components/Slider";
import { UpscaleParamsState } from "./schema";

interface UpscaleParamsProps {
  state: UpscaleParamsState;
}

const FactorParam = ({ state }: UpscaleParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Factor">
      <Select value={snap.factor.toString()} onChange={(x) => (state.factor = Number(x))}>
        <SelectItem text="2x" value="2" />
        <SelectItem text="4x" value="4" />
      </Select>
    </FormLabel>
  );
};

const DenoisingParam = ({ state }: UpscaleParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Denoising">
      <Slider value={snap.denoising} onChange={(x) => (state.denoising = x)} />
    </FormLabel>
  );
};

const BlendParam = ({ state }: UpscaleParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Blend">
      <Slider value={snap.blend} onChange={(x) => (state.blend = x)} />
    </FormLabel>
  );
};

export const UpscaleParams = ({ state }: UpscaleParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer
      label="Upscale"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <FactorParam state={state} />
      <DenoisingParam state={state} />
      <BlendParam state={state} />
    </CollapsibleContainer>
  );
};
