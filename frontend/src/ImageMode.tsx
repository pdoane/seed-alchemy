import { ImageGenerationPanel } from "./ImageGenerationPanel";
import { ImageViewer } from "./ImageViewer";
import { SessionProgress } from "./SessionProgress";
import { ThumbnailViewer } from "./ThumbnailViewer";
import { stateSettings } from "./store";

export const ImageMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden">
        <ImageGenerationPanel state={stateSettings.generation} />
        <ImageViewer />
        <ThumbnailViewer />
      </div>
    </div>
  );
};
