import { useSnapshot } from "valtio";
import { Button } from "./components/Button";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { SpinBox } from "./components/SpinBox";
import { SeedParamsState } from "./schema";

function generateSeed(): number {
  return 1 + Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

interface SeedParamsProps {
  state: SeedParamsState;
}

const SeedParam = ({ state }: SeedParamsProps) => {
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

export const SeedParams = ({ state }: SeedParamsProps) => {
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
