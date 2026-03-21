import { Dialog, Flex, Text, Button, TextField } from "@radix-ui/themes";
import { useState, useEffect } from "react";

interface PasswordDialogProps {
  open: boolean;
  username: string;
  onOpenChange: (open: boolean) => void;
  onSubmit: (password: string) => Promise<boolean>;
}

export function PasswordDialog({
  open,
  username,
  onOpenChange,
  onSubmit,
}: PasswordDialogProps) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setPassword("");
      setError(null);
      setVerifying(false);
    }
  }, [open]);

  const handleSubmit = async () => {
    if (!password) return;

    setVerifying(true);
    setError(null);

    const valid = await onSubmit(password);
    setVerifying(false);

    if (!valid) {
      setError("Incorrect password");
      setPassword("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && password && !verifying) {
      handleSubmit();
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="400px">
        <Dialog.Title>Enter Password</Dialog.Title>
        <Dialog.Description size="2" color="gray">
          Enter password for {username}
        </Dialog.Description>

        <Flex direction="column" gap="3" mt="4">
          <TextField.Root
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
            autoFocus
          />

          {error && (
            <Text size="2" color="red">
              {error}
            </Text>
          )}

          <Flex gap="3" justify="end">
            <Dialog.Close>
              <Button variant="soft" color="gray" disabled={verifying}>
                Cancel
              </Button>
            </Dialog.Close>
            <Button onClick={handleSubmit} disabled={!password || verifying}>
              {verifying ? "Verifying..." : "Sign In"}
            </Button>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
