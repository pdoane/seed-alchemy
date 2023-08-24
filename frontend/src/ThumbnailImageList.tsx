import { MouseEvent, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { useSnapshot } from "valtio";
import { ThumbnailMenu } from "./ThumbnailMenu";
import { ContextMenu } from "./components/Menu";
import { useImages } from "./queries";
import { stateSession, stateSettings, stateSystem } from "./store";
import { cx } from "./util/classNameUtil";

export const ThumbnailImageList = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSettings = useSnapshot(stateSettings);
  const snapSystem = useSnapshot(stateSystem);
  const queryImages = useImages(snapSettings.collection);
  const imagePath = queryImages.data?.[snapSession.selectedIndex ?? -1] ?? null;
  const [contextMenuPoint, setContextMenuPoint] = useState<DOMPoint | null>(null);

  function updateSelection(delta: number) {
    if (queryImages.data && stateSession.selectedIndex !== null) {
      const newIndex = stateSession.selectedIndex + delta;
      if (newIndex >= 0 && newIndex < queryImages.data.length) {
        stateSession.selectedIndex = newIndex;
      }
    }
  }

  useHotkeys("left", () => updateSelection(-1), [snapSession.selectedIndex, queryImages]);
  useHotkeys("right", () => updateSelection(1), [snapSession.selectedIndex, queryImages]);

  function handleContextMenu(event: MouseEvent<HTMLDivElement>, index: number): void {
    event.preventDefault();
    stateSession.selectedIndex = index;
    setContextMenuPoint(new DOMPoint(event.clientX, event.clientY));
  }

  return (
    <div
      className="flex-grow px-2 pb-2 pt-1 gap-3 grid content-start overflow-y-scroll"
      style={{
        gridTemplateColumns: "repeat(auto-fit, minmax(96px, 1fr))",
      }}
    >
      {queryImages.data?.map((str, index) => (
        <div
          key={str}
          className={cx(
            "flex p-0.5 aspect-square items-center justify-center relative",
            snapSession.selectedIndex == index ? "bg-slate-800" : "",
            "hover:outline hover:outline-zinc-500"
          )}
          onClick={() => (stateSession.selectedIndex = index)}
          onContextMenu={(e) => handleContextMenu(e, index)}
        >
          <img
            src={`thumbnails/${snapSystem.user}/${str}`}
            loading="lazy"
            className="max-h-full max-w-full select-none"
          />
          {snapSession.selectedIndex == index && (
            <svg className="absolute inset-0" viewBox="0 0 100 100" preserveAspectRatio="none">
              <polyline
                points="30,50 45,65 75,35"
                stroke="lime"
                strokeWidth="8"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      ))}
      {contextMenuPoint && imagePath && (
        <ContextMenu point={contextMenuPoint} onClose={() => setContextMenuPoint(null)}>
          <ThumbnailMenu imagePath={imagePath} />
        </ContextMenu>
      )}
    </div>
  );
};
