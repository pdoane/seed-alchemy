import { useSnapshot } from "valtio";
import { SourceImage } from "./SourceImage";
import { CollapsibleContainer } from "./components/Container";
import { Switch } from "./components/Switch";
import { InpaintParamsState } from "./schema";

interface ImageInpaintParamsProps {
  state: InpaintParamsState;
}

const SourceParam = ({ state }: ImageInpaintParamsProps) => {
  const snap = useSnapshot(state, { sync: true });

  return <SourceImage label="Source" value={snap.source} onChange={(x) => (state.source = x)} />;
};

const UseAlphaChannelParam = ({ state }: ImageInpaintParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <div className="flex justify-between">
      <label className="text-zinc-300">Use Alpha Channel</label>
      <Switch value={snap.useAlphaChannel} onChange={(x) => (state.useAlphaChannel = x)} />
    </div>
  );
};

const InvertMaskParam = ({ state }: ImageInpaintParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <div className="flex justify-between">
      <label className="text-zinc-300">Invert Mask</label>
      <Switch value={snap.invertMask} onChange={(x) => (state.invertMask = x)} />
    </div>
  );
};

export const ImageInpaintParams = ({ state }: ImageInpaintParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer
      label="Inpaint"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <SourceParam state={state} />
      <UseAlphaChannelParam state={state} />
      <InvertMaskParam state={state} />
    </CollapsibleContainer>
  );
};
