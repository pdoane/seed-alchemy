import { FaImage, FaQuoteLeft, FaSeedling, FaShare, FaStarOfLife, FaTrash } from "react-icons/fa";
import { MoveToMenu } from "./MoveToMenu";
import { SetAsSourceMenu } from "./SetAsSourceMenu";
import { Menu, MenuItem, MenuSeparator } from "./components/Menu";
import { toast } from "./components/Toast";
import { useRevealPath } from "./mutations";
import { useMetadata } from "./queries";
import { updateAll, updatePrompt, updateSeed, updateSourceImages } from "./requestUtil";
import { stateSession, stateSettings } from "./store";

interface CanvasGeneratedImageMenuProps {
  imagePath: string;
}

export const CanvasGeneratedImageMenu = ({ imagePath }: CanvasGeneratedImageMenuProps) => {
  const queryMetadata = useMetadata(imagePath);
  const postReveal = useRevealPath(imagePath);

  function handleUsePromptClick(): void {
    if (queryMetadata.data) {
      updatePrompt(stateSettings.generation, queryMetadata.data);
      toast("Prompt set");
    }
  }

  function handleUseSeedClick(): void {
    if (queryMetadata.data) {
      updateSeed(stateSettings.generation, queryMetadata.data);
      toast("Seed set");
    }
  }

  function handleUseSourceImagesClick(): void {
    if (queryMetadata.data) {
      updateSourceImages(stateSettings.generation, queryMetadata.data);
      toast("Source images set");
    }
  }

  function handleUseAllClick(): void {
    if (queryMetadata.data) {
      updateAll(stateSettings.generation, queryMetadata.data);
      toast("All parameters set");
    }
  }

  function handleDeleteClick(): void {
    stateSession.dialog = "delete";
    stateSession.deleteImagePath = imagePath;
  }

  function handleRevealClick(): void {
    postReveal.mutate();
  }

  return (
    <Menu>
      <MenuItem icon={FaShare} text="Set as Source">
        <SetAsSourceMenu imagePath={imagePath} />
      </MenuItem>
      <MenuSeparator />
      <MenuItem
        icon={FaQuoteLeft}
        text="Use Prompt"
        disabled={!queryMetadata.isSuccess}
        onClick={handleUsePromptClick}
      />
      <MenuItem icon={FaSeedling} text="Use Seed" disabled={!queryMetadata.isSuccess} onClick={handleUseSeedClick} />
      <MenuItem
        icon={FaImage}
        text="Use Source Images"
        disabled={!queryMetadata.isSuccess}
        onClick={handleUseSourceImagesClick}
      />
      <MenuItem icon={FaStarOfLife} text="Use All" disabled={!queryMetadata.isSuccess} onClick={handleUseAllClick} />
      <MenuSeparator />
      <MenuItem text="Move To">
        <MoveToMenu imagePath={imagePath} />
      </MenuItem>
      <MenuItem icon={FaTrash} text="Delete" onClick={handleDeleteClick} />
      <MenuSeparator />
      <MenuItem text="Reveal in Finder" onClick={handleRevealClick} />
    </Menu>
  );
};
