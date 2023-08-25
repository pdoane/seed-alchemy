import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Slider } from "./components/Slider";
import { HighResParamsState } from "./schema";

interface ImageHighResParamsProps {
  state: HighResParamsState;
}

const FactorParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Factor">
      <Slider value={snap.factor} onChange={(x) => (state.factor = x)} min={1.0} max={2.0} />
    </FormLabel>
  );
};

const StepsParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Steps">
      <Slider value={snap.steps} onChange={(x) => (state.steps = x)} min={1} max={200} step={1} decimals={0} />
    </FormLabel>
  );
};

const CfgScaleParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="CFG Scale">
      <Slider value={snap.cfgScale} onChange={(x) => (state.cfgScale = x)} min={1} max={50} step={0.5} />
    </FormLabel>
  );
};

const NoiseParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Noise">
      <Slider value={snap.noise} onChange={(x) => (state.noise = x)} />
    </FormLabel>
  );
};

export const ImageHighResParams = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer
      label="High Resolution"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <FactorParam state={state} />
      <StepsParam state={state} />
      <CfgScaleParam state={state} />
      <NoiseParam state={state} />
    </CollapsibleContainer>
  );
};
