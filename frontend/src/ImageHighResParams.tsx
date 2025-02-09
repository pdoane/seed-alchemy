import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Slider } from "./components/Slider";
import { SpinBox } from "./components/SpinBox";
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

const NoiseParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Noise">
      <Slider value={snap.noise} onChange={(x) => (state.noise = x)} />
    </FormLabel>
  );
};

const StepsParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Steps">
      <SpinBox value={snap.steps} onChange={(x) => (state.steps = x)} min={1} max={100} />
    </FormLabel>
  );
};

const CfgScaleParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="CFG Scale">
      <SpinBox value={snap.cfgScale} onChange={(x) => (state.cfgScale = x)} min={1.0} max={200} step={0.5} />
    </FormLabel>
  );
};

const ClipSkipParam = ({ state }: ImageHighResParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Clip Skip">
      <SpinBox value={snap.clipSkip} onChange={(x) => (state.clipSkip = x)} min={0} max={10} />
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
      <div className="flex space-x-3">
        <StepsParam state={state} />
        <CfgScaleParam state={state} />
        <ClipSkipParam state={state} />
      </div>
      <FactorParam state={state} />
      <NoiseParam state={state} />
    </CollapsibleContainer>
  );
};
