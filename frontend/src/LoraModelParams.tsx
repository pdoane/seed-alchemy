import { useSnapshot } from "valtio";
import { FormLabel } from "./components/FormLabel";
import { Slider } from "./components/Slider";
import { LoraModelParamsState } from "./schema";
import { IconButton } from "./components/Button";
import { FaTrash } from "react-icons/fa";

interface LoraModelParamsProps {
  state: LoraModelParamsState;
  onDelete: () => void;
}

export const LoraModelParams = ({ state, onDelete }: LoraModelParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label={snap.model}>
      <div className="flex space-x-1">
        <Slider value={snap.weight} onChange={(x) => (state.weight = x)} min={-1.0} max={2.0} />
        <IconButton icon={FaTrash} onClick={onDelete} />
      </div>
    </FormLabel>
  );
};
