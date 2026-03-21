import {
  DropdownMenu,
  IconButton,
  Text,
  Flex,
  Separator,
  Tooltip,
} from "@radix-ui/themes";
import { UserIcon, PlusIcon, LockSimpleIcon } from "@phosphor-icons/react";
import { useAppStore } from "../store/appStore";

interface UserMenuProps {
  onCreateUser: () => void;
  onSelectUser: (username: string, hasPassword: boolean) => void;
}

export function UserMenu({ onCreateUser, onSelectUser }: UserMenuProps) {
  const { currentUser, users } = useAppStore();

  return (
    <DropdownMenu.Root>
      <Tooltip content={currentUser} side="right">
        <DropdownMenu.Trigger>
          <IconButton size="3" variant="soft" color="gray" aria-label="User">
            <UserIcon size={24} weight="fill" />
          </IconButton>
        </DropdownMenu.Trigger>
      </Tooltip>
      <DropdownMenu.Content side="right" align="end">
        {/* Current user indicator */}
        <DropdownMenu.Label>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Signed in as
            </Text>
            <Text size="2" weight="bold">
              {currentUser}
            </Text>
          </Flex>
        </DropdownMenu.Label>

        <Separator size="4" my="1" />

        {/* User list */}
        {users.map((user) => (
          <DropdownMenu.Item
            key={user.username}
            onSelect={() => onSelectUser(user.username, user.hasPassword)}
            disabled={user.username === currentUser}
          >
            <Flex align="center" justify="between" style={{ width: "100%" }}>
              <Text>{user.username}</Text>
              {user.hasPassword && (
                <LockSimpleIcon size={14} style={{ marginLeft: "8px" }} />
              )}
            </Flex>
          </DropdownMenu.Item>
        ))}

        <Separator size="4" my="1" />

        {/* Create user option */}
        <DropdownMenu.Item onSelect={onCreateUser}>
          <Flex align="center" gap="2">
            <PlusIcon size={16} />
            Create User
          </Flex>
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}
