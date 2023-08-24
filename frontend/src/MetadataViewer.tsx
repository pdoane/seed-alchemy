import { JsonDisplay } from "./components/JsonDisplay";
import { useImageInfo } from "./queries";

interface MetadataViewerProps {
  path: string;
}

export const MetadataViewer = ({ path }: MetadataViewerProps) => {
  const queryImageInfo = useImageInfo(path);
  return <JsonDisplay data={queryImageInfo.data} />;
};
