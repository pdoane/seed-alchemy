import { useSnapshot } from "valtio";
import { ImageToolbar } from "./ImageToolbar";
import { MetadataViewer } from "./MetadataViewer";
import { useImages } from "./queries";
import { stateSession, stateSettings, stateSystem } from "./store";

export const ImageViewer = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSettings = useSnapshot(stateSettings);
  const snapSystem = useSnapshot(stateSystem);
  const queryImages = useImages(snapSettings.collection);
  const imagePath = queryImages.data?.[snapSession.selectedIndex ?? -1] ?? null;

  const imageUrl =
    snapSettings.showPreview && snapSession.previewUrl && snapSession.generatorId == null
      ? snapSession.previewUrl
      : imagePath
      ? `images/${snapSystem.user}/${imagePath}`
      : null;

  return (
    <div className="flex w-full h-full flex-shrink bg-black ">
      <div className="flex w-full flex-col space-y-2">
        <ImageToolbar />
        <div className="relative w-full h-full">
          {imageUrl && (
            <div className="absolute flex w-full h-full items-center justify-center">
              <img className="max-h-full max-w-full select-none" src={imageUrl} />
            </div>
          )}
          {snapSettings.showMetadata && (
            <div className="absolute w-full h-full px-2 overflow-y-scroll bg-black bg-opacity-75">
              {imagePath && <MetadataViewer path={imagePath} />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
