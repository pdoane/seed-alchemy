import {
  Dialog,
  Flex,
  Text,
  Button,
  TextField,
  Checkbox,
} from "@radix-ui/themes";
import { useState, useEffect } from "react";

interface CreateUserDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreateUser: (username: string, password?: string) => Promise<void>;
}

// Username validation: alphanumeric + underscore + hyphen, 1-32 chars
const USERNAME_REGEX = /^[a-zA-Z0-9_-]{1,32}$/;

export function CreateUserDialog({
  open,
  onOpenChange,
  onCreateUser,
}: CreateUserDialogProps) {
  const [username, setUsername] = useState("");
  const [usePassword, setUsePassword] = useState(false);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setUsername("");
      setUsePassword(false);
      setPassword("");
      setConfirmPassword("");
      setError(null);
      setCreating(false);
    }
  }, [open]);

  const validateUsername = (name: string): string | null => {
    if (!name) return "Username is required";
    if (!USERNAME_REGEX.test(name)) {
      return "Username must be 1-32 characters: letters, numbers, underscores, or hyphens";
    }
    return null;
  };

  const handleSubmit = async () => {
    setError(null);

    // Validate username
    const usernameError = validateUsername(username);
    if (usernameError) {
      setError(usernameError);
      return;
    }

    // Validate password if enabled
    if (usePassword) {
      if (!password) {
        setError("Password is required");
        return;
      }
      if (password !== confirmPassword) {
        setError("Passwords do not match");
        return;
      }
    }

    setCreating(true);

    try {
      await onCreateUser(username, usePassword ? password : undefined);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create user");
    } finally {
      setCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && username && !creating) {
      if (!usePassword || (password && password === confirmPassword)) {
        handleSubmit();
      }
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="400px">
        <Dialog.Title>Create User</Dialog.Title>
        <Dialog.Description size="2" color="gray">
          Create a new user account
        </Dialog.Description>

        <Flex direction="column" gap="3" mt="4">
          <Flex direction="column" gap="1">
            <Text size="2" weight="medium">
              Username
            </Text>
            <TextField.Root
              placeholder="Enter username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
            />
          </Flex>

          <Text
            as="label"
            size="2"
            style={{ display: "flex", alignItems: "center", gap: "8px" }}
          >
            <Checkbox
              checked={usePassword}
              onCheckedChange={(checked) => setUsePassword(checked === true)}
            />
            Protect with password
          </Text>

          {usePassword && (
            <>
              <Flex direction="column" gap="1">
                <Text size="2" weight="medium">
                  Password
                </Text>
                <TextField.Root
                  type="password"
                  placeholder="Enter password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </Flex>

              <Flex direction="column" gap="1">
                <Text size="2" weight="medium">
                  Confirm Password
                </Text>
                <TextField.Root
                  type="password"
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </Flex>
            </>
          )}

          {error && (
            <Text size="2" color="red">
              {error}
            </Text>
          )}

          <Flex gap="3" justify="end" mt="2">
            <Dialog.Close>
              <Button variant="soft" color="gray" disabled={creating}>
                Cancel
              </Button>
            </Dialog.Close>
            <Button onClick={handleSubmit} disabled={!username || creating}>
              {creating ? "Creating..." : "Create User"}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
