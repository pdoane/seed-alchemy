import { Menu, MenuItem } from "./components/Menu";
import { useCollections } from "./queries";
import { useMoveImage } from "./mutations";

interface MoveToMenuProps {
  imagePath: string;
}

export const MoveToMenu = ({ imagePath }: MoveToMenuProps) => {
  const queryCollections = useCollections();
  const postMove = useMoveImage(imagePath);

  function handleMoveToClick(dstCollection: string): void {
    postMove.mutate(dstCollection);
  }

  return (
    <Menu>
      {queryCollections.data?.map((str, _) => (
        <MenuItem key={str} text={str} onClick={() => handleMoveToClick(str)} />
      ))}
    </Menu>
  );
};
