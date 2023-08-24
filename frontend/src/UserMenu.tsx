import { useSnapshot } from "valtio";
import { Menu, MenuItem } from "./components/Menu";
import { QueryStatusMenuItem } from "./components/QueryStatus";
import { useUsers } from "./queries";
import { stateSession, stateSystem } from "./store";
import { FaCheck } from "react-icons/fa";

export const UserMenu = () => {
  const snapSystem = useSnapshot(stateSystem);
  const queryUsers = useUsers();

  if (!queryUsers.isSuccess) {
    return (
      <Menu>
        <QueryStatusMenuItem result={queryUsers} />
      </Menu>
    );
  }

  function handleUserClick(user: string): void {
    stateSystem.user = user;
    stateSession.historyStack = [];
    stateSession.historyStackIndex = -1;
  }

  return (
    <Menu>
      {queryUsers.data?.map((str, _) => (
        <MenuItem
          key={str}
          text={str}
          icon={str == snapSystem.user ? FaCheck : null}
          onClick={() => handleUserClick(str)}
        />
      ))}
    </Menu>
  );
};
