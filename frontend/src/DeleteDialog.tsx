import { useSnapshot } from "valtio";
import { stateSession, stateSystem } from "./store";
import { Modal } from "./components/Modal";
import { Button } from "./components/Button";
import { useDeleteImage } from "./mutations";
import { useEffect, useRef } from "react";

export const DeleteDialog = () => {
  const snapSession = useSnapshot(stateSession);
  const snapSystem = useSnapshot(stateSystem);
  const defaultRef = useRef<HTMLButtonElement | null>(null);
  const postDelete = useDeleteImage(snapSession.deleteImagePath);

  function close(): void {
    stateSession.dialog = null;
  }

  useEffect(() => {
    if (defaultRef.current) {
      defaultRef.current.focus();
    }
  }, []);

  function handleDeleteClick(): void {
    postDelete.mutate();
    close();
  }

  return (
    <Modal onClose={close}>
      <div className="flex flex-col space-y-2">
        <span className="font-bold text-lg">Confirm Delete</span>
        <span>Are you sure you want to delete this image?</span>
        <div className="flex aspect-square w-96 h-96 items-center justify-center ">
          <img src={`images/${snapSystem.user}/${snapSession.deleteImagePath}`} className="max-h-full max-w-full" />
        </div>
        <div className="w-full flex justify-end space-x-3">
          <Button onClick={close}>Cancel</Button>
          <Button variant="primary" ref={defaultRef} onClick={handleDeleteClick}>
            Delete
          </Button>
        </div>
      </div>
    </Modal>
  );
};
