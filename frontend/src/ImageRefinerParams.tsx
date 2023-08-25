import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Slider } from "./components/Slider";
import { Switch } from "./components/Switch";
import { RefinerParamsState } from "./schema";

interface ImageRefinerParamsProps {
  state: RefinerParamsState;
  baseModelType: string | undefined;
}

const EnsembleModeParam = ({ state }: ImageRefinerParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <div className="flex justify-between">
      <label className="text-zinc-300">Ensemble Mode</label>
      <Switch value={snap.ensembleMode} onChange={(x) => (state.ensembleMode = x)} />
    </div>
  );
};

const CfgScaleParam = ({ state }: ImageRefinerParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="CFG Scale">
      <Slider value={snap.cfgScale} onChange={(x) => (state.cfgScale = x)} min={1} max={50} step={0.5} />
    </FormLabel>
  );
};

const HighNoiseEndParam = ({ state }: ImageRefinerParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="High Noise End">
      <Slider value={snap.highNoiseEnd} onChange={(x) => (state.highNoiseEnd = x)} />
    </FormLabel>
  );
};

const StepsParam = ({ state }: ImageRefinerParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Steps">
      <Slider value={snap.steps} onChange={(x) => (state.steps = x)} min={1} max={200} step={1} decimals={0} />
    </FormLabel>
  );
};

const NoiseParam = ({ state }: ImageRefinerParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Noise">
      <Slider value={snap.noise} onChange={(x) => (state.noise = x)} />
    </FormLabel>
  );
};

export const ImageRefinerParams = (props: ImageRefinerParamsProps) => {
  const { state, baseModelType } = props;
  const snap = useSnapshot(state);

  if (baseModelType !== "sdxl") return <></>;

  return (
    <CollapsibleContainer
      label="Refiner"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <EnsembleModeParam {...props} />
      <CfgScaleParam {...props} />
      {snap.ensembleMode && <HighNoiseEndParam {...props} />}
      {!snap.ensembleMode && <StepsParam {...props} />}
      {!snap.ensembleMode && <NoiseParam {...props} />}
    </CollapsibleContainer>
  );
};
