import { useSnapshot } from "valtio";
import { ThumbnailImageList } from "./ThumbnailImageList";
import { Select, SelectItem } from "./components/Select";
import { useCollections } from "./queries";
import { stateSession, stateSettings } from "./store";

export const ThumbnailViewer = () => {
  const snapSettings = useSnapshot(stateSettings);
  const queryCollections = useCollections();

  function handleChange(value: string): void {
    stateSettings.collection = value;
    stateSession.selectedIndex = 0;
  }

  return (
    <div className="w-72 h-full flex-shrink-0 flex flex-col space-y-2 bg-zinc-900">
      <div className="px-2 pt-2">
        <Select value={snapSettings.collection} onChange={handleChange} status={queryCollections.status}>
          {queryCollections.data?.map((str, _) => (
            <SelectItem key={str} text={str} value={str} />
          ))}
        </Select>
      </div>
      <ThumbnailImageList />
    </div>
  );
};
