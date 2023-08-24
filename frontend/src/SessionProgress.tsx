import { useSnapshot } from "valtio";
import { ProgressBar } from "./components/ProgressBar";
import { stateSession } from "./store";

export const SessionProgress = () => {
  const snapSession = useSnapshot(stateSession);

  return <ProgressBar value={snapSession.progressAmount} max={100} />;
};
