import { useState } from "react";
import { Box, Flex, IconButton, Text, Tooltip } from "@radix-ui/themes";
import {
  ImageSquareIcon,
  StackIcon,
  PaintBrushIcon,
  ImagesIcon,
  CubeIcon,
  GearIcon,
} from "@phosphor-icons/react";
import { useAppStore, type AppMode } from "../store/appStore";
import { api } from "../api/client";
import { switchUser } from "../store/persistence";
import { UserMenu } from "./UserMenu";
import { PasswordDialog } from "./PasswordDialog";
import { CreateUserDialog } from "./CreateUserDialog";
import { SettingsDialog } from "./SettingsDialog";

const modes: { id: AppMode; label: string; icon: React.ReactNode }[] = [
  {
    id: "image",
    label: "Image",
    icon: <ImageSquareIcon size={24} weight="fill" />,
  },
  {
    id: "document",
    label: "Document",
    icon: <StackIcon size={24} weight="fill" />,
  },
  {
    id: "canvas",
    label: "Canvas",
    icon: <PaintBrushIcon size={24} weight="fill" />,
  },
  {
    id: "gallery",
    label: "Gallery",
    icon: <ImagesIcon size={24} weight="fill" />,
  },
  { id: "models", label: "Models", icon: <CubeIcon size={24} weight="fill" /> },
];

export function Sidebar() {
  const { mode, setMode, loadUsers } = useAppStore();

  // Dialog state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [createUserOpen, setCreateUserOpen] = useState(false);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [pendingUser, setPendingUser] = useState<string | null>(null);

  const handleSelectUser = async (username: string, hasPassword: boolean) => {
    if (hasPassword) {
      setPendingUser(username);
      setPasswordDialogOpen(true);
    } else {
      await switchUser(username);
    }
  };

  const handlePasswordSubmit = async (password: string): Promise<boolean> => {
    if (!pendingUser) return false;

    const valid = await api.verifyPassword(pendingUser, password);
    if (valid) {
      await switchUser(pendingUser);
      setPasswordDialogOpen(false);
      setPendingUser(null);
    }
    return valid;
  };

  const handleCreateUser = async (
    username: string,
    password?: string
  ): Promise<void> => {
    await api.createUser(username, password);
    await loadUsers();
    await switchUser(username);
  };

  return (
    <Box asChild width="64px" className="border-r border-[var(--gray-6)]">
      <aside>
        <Flex
          direction="column"
          align="center"
          justify="between"
          py="4"
          height="100%"
        >
          {/* Top section: Logo + mode buttons */}
          <Flex direction="column" align="center" gap="2">
            <Text size="5" weight="bold" color="violet" mb="4">
              SA
            </Text>
            {modes.map((m) => (
              <Tooltip key={m.id} content={m.label} side="right">
                <IconButton
                  size="3"
                  variant={mode === m.id ? "solid" : "soft"}
                  color={mode === m.id ? "violet" : "gray"}
                  onClick={() => setMode(m.id)}
                  aria-label={m.label}
                >
                  {m.icon}
                </IconButton>
              </Tooltip>
            ))}
          </Flex>

          {/* Bottom section: Settings + User */}
          <Flex direction="column" align="center" gap="2">
            <Tooltip content="Settings" side="right">
              <IconButton
                size="3"
                variant="soft"
                color="gray"
                onClick={() => setSettingsOpen(true)}
                aria-label="Settings"
              >
                <GearIcon size={24} weight="fill" />
              </IconButton>
            </Tooltip>

            <UserMenu
              onCreateUser={() => setCreateUserOpen(true)}
              onSelectUser={handleSelectUser}
            />
          </Flex>
        </Flex>

        {/* Dialogs */}
        <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
        <CreateUserDialog
          open={createUserOpen}
          onOpenChange={setCreateUserOpen}
          onCreateUser={handleCreateUser}
        />
        <PasswordDialog
          open={passwordDialogOpen}
          username={pendingUser || ""}
          onOpenChange={setPasswordDialogOpen}
          onSubmit={handlePasswordSubmit}
        />
      </aside>
    </Box>
  );
}
