import { Menu, MenuItem } from "./components/Menu";
import { stateSession, stateSettings } from "./store";
import { SetAsSourceMenu } from "./SetAsSourceMenu";
import { FaCompass, FaShare } from "react-icons/fa";
import { dirName } from "./util/pathUtil";
import { useImages } from "./queries";

interface SourceImageMenuProps {
  imagePath: string;
}

export const SourceImageMenu = ({ imagePath }: SourceImageMenuProps) => {
  const collection = dirName(imagePath);
  const queryImages = useImages(collection);

  function handleLocateImageClick(): void {
    if (queryImages.data) {
      const index = queryImages.data.findIndex((x) => x == imagePath);
      if (index >= 0) {
        stateSession.selectedIndex = index;
        stateSettings.collection = collection;
      }
    }
  }

  return (
    <Menu>
      <MenuItem
        icon={FaCompass}
        text="Locate Image"
        disabled={!queryImages.isSuccess}
        onClick={handleLocateImageClick}
      />
      <MenuItem icon={FaShare} text="Set as Source">
        <SetAsSourceMenu imagePath={imagePath} />
      </MenuItem>
    </Menu>
  );
};
