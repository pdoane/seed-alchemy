import {
  Dialog,
  Flex,
  Text,
  Button,
  AlertDialog,
  Separator,
} from "@radix-ui/themes";
import { useState } from "react";
import { useAppStore } from "../store/appStore";
import { api } from "../api/client";
import { switchUser } from "../store/persistence";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { currentUser, users, loadUsers } = useAppStore();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDeleteUser = async () => {
    setDeleting(true);
    setError(null);

    try {
      await api.deleteUser(currentUser);

      // Reload users list
      await loadUsers();

      // Find another user to switch to, or create default
      const remainingUsers = useAppStore.getState().users;
      const firstUser = remainingUsers[0];
      if (firstUser) {
        await switchUser(firstUser.username);
      } else {
        // Create a default user if no users left
        await api.createUser("default");
        await loadUsers();
        await switchUser("default");
      }

      setShowDeleteConfirm(false);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete user");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="500px">
        <Dialog.Title>Settings</Dialog.Title>

        <Flex direction="column" gap="4" mt="4">
          {/* Account section */}
          <Flex direction="column" gap="2">
            <Text size="2" weight="bold">
              Account
            </Text>
            <Text size="2" color="gray">
              Signed in as <Text weight="medium">{currentUser}</Text>
            </Text>
          </Flex>

          <Separator size="4" />

          {/* Danger zone */}
          <Flex direction="column" gap="2">
            <Text size="2" weight="bold" color="red">
              Danger Zone
            </Text>
            <Text size="2" color="gray">
              Permanently delete your account and all associated data.
            </Text>

            {error && (
              <Text size="2" color="red">
                {error}
              </Text>
            )}

            <AlertDialog.Root
              open={showDeleteConfirm}
              onOpenChange={setShowDeleteConfirm}
            >
              <AlertDialog.Trigger>
                <Button
                  color="red"
                  variant="soft"
                  disabled={users.length <= 1}
                  style={{ width: "fit-content" }}
                >
                  Delete Account
                </Button>
              </AlertDialog.Trigger>
              <AlertDialog.Content maxWidth="450px">
                <AlertDialog.Title>Delete Account?</AlertDialog.Title>
                <AlertDialog.Description size="2">
                  This will permanently delete all your images and settings for
                  user &quot;{currentUser}&quot;. This action cannot be undone.
                </AlertDialog.Description>
                <Flex gap="3" justify="end" mt="4">
                  <AlertDialog.Cancel>
                    <Button variant="soft" color="gray" disabled={deleting}>
                      Cancel
                    </Button>
                  </AlertDialog.Cancel>
                  <AlertDialog.Action>
                    <Button
                      color="red"
                      onClick={handleDeleteUser}
                      disabled={deleting}
                    >
                      {deleting ? "Deleting..." : "Delete Account"}
                    </Button>
                  </AlertDialog.Action>
                </Flex>
              </AlertDialog.Content>
            </AlertDialog.Root>

            {users.length <= 1 && (
              <Text size="1" color="gray">
                Cannot delete the last user account.
              </Text>
            )}
          </Flex>
        </Flex>

        <Flex justify="end" mt="4">
          <Dialog.Close>
            <Button variant="soft" color="gray">
              Close
            </Button>
          </Dialog.Close>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
