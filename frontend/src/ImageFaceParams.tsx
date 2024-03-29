import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { Slider } from "./components/Slider";
import { FaceParamsState } from "./schema";

interface ImageFaceParamsProps {
  state: FaceParamsState;
}

const BlendParam = ({ state }: ImageFaceParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Blend">
      <Slider value={snap.blend} onChange={(x) => (state.blend = x)} />
    </FormLabel>
  );
};

export const ImageFaceParams = ({ state }: ImageFaceParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer
      label="Face Restoration"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <BlendParam state={state} />
    </CollapsibleContainer>
  );
};
