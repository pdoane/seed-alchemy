import { SessionProgress } from "./SessionProgress";

export const GalleryMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden items-center justify-center">Gallery Mode - Coming Soon</div>
    </div>
  );
};
