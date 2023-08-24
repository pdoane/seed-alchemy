import { FaBrush, FaComments, FaImage, FaImages, FaQuestion, FaUserCircle } from "react-icons/fa";
import { FaGear } from "react-icons/fa6";
import { useSnapshot } from "valtio";
import { UserMenu } from "./UserMenu";
import { ModeButton, ToolbarButton } from "./components/Button";
import { RadioGroup } from "./components/RadioGroup";
import { stateSession, stateSystem } from "./store";

export const AppToolbar = () => {
  const snapSystem = useSnapshot(stateSystem);

  function onSettingsClick(): void {
    stateSession.dialog = "settings";
  }

  return (
    <div className="h-full flex flex-col justify-between bg-zinc-800">
      <div>
        <RadioGroup value={snapSystem.mode} onChange={(x) => (stateSystem.mode = x)}>
          <ModeButton value="image" icon={FaImage} />
          <ModeButton value="canvas" icon={FaBrush} />
          <ModeButton value="gallery" icon={FaImages} />
          <ModeButton value="prompt" icon={FaComments} />
          <ModeButton value="interrogate" icon={FaQuestion} />
        </RadioGroup>
      </div>
      <div>
        <ToolbarButton icon={FaUserCircle} menu={<UserMenu />} placement="right" />
        <ToolbarButton icon={FaGear} onClick={onSettingsClick} />
      </div>
    </div>
  );
};
