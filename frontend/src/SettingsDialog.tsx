import { Checkbox } from "./components/Button";
import { useSnapshot } from "valtio";
import { stateSession, stateSettings } from "./store";
import { Modal } from "./components/Modal";

export const SettingsDialog = () => {
  const snapSettings = useSnapshot(stateSettings);

  return (
    <Modal onClose={() => (stateSession.dialog = null)}>
      <span className="font-bold text-lg text-white">Settings</span>
      <Checkbox
        label="Safety Checker"
        value={snapSettings.safetyChecker}
        onChange={(x) => (stateSettings.safetyChecker = x)}
      />
    </Modal>
  );
};
