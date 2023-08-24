import { createContext, MouseEvent, ReactNode, useContext, useState } from "react";
import { AiOutlineClose, AiOutlinePlus } from "react-icons/ai";
import { cx } from "../util/classNameUtil";

type TabsContextProps = {
  activeTab: number;
  onTabChange?: (index: number) => void;
};

type TabsProps = {
  activeTab: number;
  onTabChange?: (index: number) => void;
  children: ReactNode;
};

type TabListProps = {
  children: ReactNode;
};

type TabProps = {
  children: string;
  index: number;
  hasClose?: boolean;
  onClose?: (index: number) => void;
};

type TabNewButtonProps = {
  onClick?: () => void;
};

type TabPanelsProps = {
  children: ReactNode;
};

type TabPanelProps = {
  children: ReactNode;
  index: number;
};

export const TabsContext = createContext<TabsContextProps>({ activeTab: 0 });

export const Tabs = ({ activeTab, onTabChange, children }: TabsProps) => {
  return (
    <TabsContext.Provider value={{ activeTab, onTabChange }}>
      <div>{children}</div>
    </TabsContext.Provider>
  );
};

export const TabList = ({ children }: TabListProps) => {
  return <div className="flex overflow-hidden bg-zinc-900">{children}</div>;
};

export const Tab = ({ children, index, hasClose = true, onClose }: TabProps) => {
  const { activeTab, onTabChange } = useContext(TabsContext);
  const [isHovered, setIsHovered] = useState(false);

  const isActive = activeTab === index;

  return (
    <div
      className={cx(
        "relative flex pt-1 pb-0.5 items-center hover:bg-zinc-800 border-r-2 border-b-2 border-r-zinc-800",
        isActive ? " border-b-blue-600" : "border-b-transparent"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <button onClick={() => onTabChange?.(index)} className={cx("px-6", isActive ? "text-white" : " text-zinc-400")}>
        {children}
      </button>
      {hasClose && (
        <button
          onClick={() => onClose?.(index)}
          className={cx(
            "absolute right-1 transition-opacity duration-250",
            isHovered ? "opacity-100  text-white" : "opacity-0"
          )}
        >
          <AiOutlineClose />
        </button>
      )}
    </div>
  );
};

export const TabNewButton = ({ onClick }: TabNewButtonProps) => {
  function handleClick(event: MouseEvent<HTMLButtonElement>): void {
    if (onClick) onClick();
    event.stopPropagation();
  }

  return (
    <button
      className={cx("p-1", "bg-zinc-900 hover:bg-zinc-700", "transition duration-250 ease-in-out")}
      onClick={handleClick}
    >
      <AiOutlinePlus />
    </button>
  );
};

export const TabPanels = ({ children }: TabPanelsProps) => {
  return <div className="pt-2">{children}</div>;
};

export const TabPanel = ({ children, index }: TabPanelProps) => {
  const { activeTab } = useContext(TabsContext);

  if (activeTab !== index) return null;

  return children;
};
