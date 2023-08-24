import { useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import { dirName } from "./util/pathUtil";
import { useImages } from "./queries";
import { stateSession, stateSettings } from "./store";

const MAX_HISTORY = 30;

export const ImageHistory = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSettings = useSnapshot(stateSettings);
  const queryImages = useImages(snapSettings.collection);

  const [pendingCollection, setPendingCollection] = useState<string | null>(null);
  const queryPendingImages = useImages(pendingCollection);

  // selectedIndex change -> imagePath && historyStack
  useEffect(() => {
    const imagePath = queryImages.data?.[stateSession.selectedIndex ?? -1] ?? null;
    if (imagePath) {
      if (stateSession.historyStack[stateSession.historyStackIndex] == imagePath) return;

      let newStack =
        stateSession.historyStackIndex < stateSession.historyStack.length - 1
          ? stateSession.historyStack.slice(0, stateSession.historyStackIndex + 1)
          : stateSession.historyStack;

      newStack = [...newStack, imagePath];
      if (newStack.length > MAX_HISTORY) {
        newStack.shift();
      } else {
        stateSession.historyStackIndex += 1;
      }

      stateSession.historyStack = newStack;
    }
  }, [snapSession.selectedIndex, queryImages.data]);

  // history change -> pending collection fetch
  useEffect(() => {
    const imagePath = stateSession.historyStack[stateSession.historyStackIndex];
    if (imagePath) {
      const collection = dirName(imagePath);
      setPendingCollection(collection);
    }
  }, [snapSession.historyStackIndex, snapSession.historyStack]);

  // pending collection fetch -> selectedIndex/collection
  useEffect(() => {
    const imagePath = stateSession.historyStack[stateSession.historyStackIndex];
    if (imagePath) {
      const collection = dirName(imagePath);
      if (pendingCollection == collection && queryPendingImages.data) {
        const index = queryPendingImages.data.findIndex((x) => x == imagePath);
        if (index >= 0) {
          stateSession.selectedIndex = index;
          stateSettings.collection = pendingCollection;
        }
      }
    }
  }, [snapSession.historyStackIndex, snapSession.historyStack, queryPendingImages.data]);

  return <></>;
};
