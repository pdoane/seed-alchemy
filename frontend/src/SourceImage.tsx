import { useState, MouseEvent } from "react";
import { TextInput } from "./components/TextInput";
import { useSnapshot } from "valtio";
import { stateSystem } from "./store";
import { ContextMenu } from "./components/Menu";
import { SourceImageMenu } from "./SourceImageMenu";
import { Image } from "./components/Image";

interface SourceImageProps {
  label?: string;
  value: string;
  onChange?: (value: string) => void;
}

export const SourceImage = ({ label, value, onChange }: SourceImageProps) => {
  const snapSystem = useSnapshot(stateSystem);
  const [contextMenuPoint, setContextMenuPoint] = useState<DOMPoint | null>(null);

  function handleContextMenu(event: MouseEvent<HTMLDivElement>): void {
    event.preventDefault();
    setContextMenuPoint(new DOMPoint(event.clientX, event.clientY));
  }

  return (
    <div className="w-full flex space-x-3 items-center">
      <TextInput className="pb-4" label={label} value={value} placeholder="Image Path" onChange={onChange} />
      <div
        className="flex aspect-square w-24 h-24 items-center justify-center border border-zinc-500 bg-zinc-950"
        onContextMenu={handleContextMenu}
      >
        <Image className="max-h-full max-w-full" src={`thumbnails/${snapSystem.user}/${value}`} />
      </div>
      {contextMenuPoint && (
        <ContextMenu point={contextMenuPoint} onClose={() => setContextMenuPoint(null)}>
          <SourceImageMenu imagePath={value} />
        </ContextMenu>
      )}
    </div>
  );
};
