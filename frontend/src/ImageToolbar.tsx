import { useHotkeys } from "react-hotkeys-hook";
import { AiOutlineSearch } from "react-icons/ai";
import {
  FaArrowLeft,
  FaArrowRight,
  FaImage,
  FaInfoCircle,
  FaQuoteLeft,
  FaSeedling,
  FaShare,
  FaStarOfLife,
  FaTrash,
} from "react-icons/fa";
import { useSnapshot } from "valtio";
import { SetAsSourceMenu } from "./SetAsSourceMenu";
import { ToolbarButton, ToolbarCheckButton } from "./components/Button";
import { toast } from "./components/Toast";
import { useImages, useMetadata } from "./queries";
import { updateAll, updatePrompt, updateSeed, updateSourceImages } from "./requestUtil";
import { stateSession, stateSettings } from "./store";

export const ImageToolbar = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSettings = useSnapshot(stateSettings);
  const queryImages = useImages(snapSettings.collection);
  const imagePath = queryImages.data?.[snapSession.selectedIndex ?? -1] ?? null;
  const queryMetadata = useMetadata(imagePath);

  function handleBackClick(): void {
    if (stateSession.historyStackIndex > 0) {
      stateSession.historyStackIndex -= 1;
    }
  }

  function handleForwardClick(): void {
    if (stateSession.historyStackIndex < stateSession.historyStack.length - 1) {
      stateSession.historyStackIndex += 1;
    }
  }

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
    if (imagePath) {
      stateSession.dialog = "delete";
      stateSession.deleteImagePath = imagePath;
    }
  }

  useHotkeys(
    "mod+backspace",
    handleDeleteClick,
    {
      enabled: () => !!imagePath,
      preventDefault: true,
      enableOnFormTags: ["input", "textarea", "select"],
    },
    [imagePath]
  );

  return (
    <div className="flex justify-center items-center space-x-4 bg-zinc-900">
      <div className="flex space-x-0.5">
        <ToolbarButton icon={FaArrowLeft} disabled={snapSession.historyStackIndex <= 0} onClick={handleBackClick} />
        <ToolbarButton
          icon={FaArrowRight}
          disabled={snapSession.historyStackIndex >= snapSession.historyStack.length - 1}
          onClick={handleForwardClick}
        />
      </div>
      <ToolbarButton
        icon={FaShare}
        disabled={!imagePath}
        menu={imagePath && <SetAsSourceMenu imagePath={imagePath} />}
      />
      <div className="flex space-x-0.5">
        <ToolbarButton icon={FaQuoteLeft} disabled={!queryMetadata.isSuccess} onClick={handleUsePromptClick} />
        <ToolbarButton icon={FaSeedling} disabled={!queryMetadata.isSuccess} onClick={handleUseSeedClick} />
        <ToolbarButton icon={FaImage} disabled={!queryMetadata.isSuccess} onClick={handleUseSourceImagesClick} />
        <ToolbarButton icon={FaStarOfLife} disabled={!queryMetadata.isSuccess} onClick={handleUseAllClick} />
      </div>
      <div className="flex space-x-0.5">
        <ToolbarCheckButton
          icon={FaInfoCircle}
          value={snapSettings.showMetadata}
          onChange={(x) => (stateSettings.showMetadata = x)}
        />
        <ToolbarCheckButton
          icon={AiOutlineSearch}
          value={snapSettings.showPreview}
          onChange={(x) => (stateSettings.showPreview = x)}
        />
      </div>
      <ToolbarButton icon={FaTrash} disabled={!imagePath} onClick={handleDeleteClick} />
    </div>
  );
};
