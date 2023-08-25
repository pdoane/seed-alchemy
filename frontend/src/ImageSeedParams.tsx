import { useSnapshot } from "valtio";
import { Button } from "./components/Button";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { SpinBox } from "./components/SpinBox";
import { generateSeed } from "./random";
import { SeedParamsState } from "./schema";

interface ImageSeedParamsProps {
  state: SeedParamsState;
}

const SeedParam = ({ state }: ImageSeedParamsProps) => {
  const snap = useSnapshot(state);

  function handleNewSeedClick(): void {
    state.seed = generateSeed();
  }

  return (
    <FormLabel label="Seed">
      <div className="flex space-x-1">
        <SpinBox className="w-full" value={snap.seed} onChange={(x) => (state.seed = x)} min={1} />
        <Button onClick={handleNewSeedClick}>New</Button>
      </div>
    </FormLabel>
  );
};

export const ImageSeedParams = ({ state }: ImageSeedParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <CollapsibleContainer
      label="Manual Seed"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <SeedParam state={state} />
    </CollapsibleContainer>
  );
};
