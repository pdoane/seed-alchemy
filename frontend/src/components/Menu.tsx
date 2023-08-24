import {
  MouseEvent,
  ReactNode,
  Ref,
  createContext,
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { IconType } from "react-icons";
import { cx } from "../util/classNameUtil";
import { FaChevronRight } from "react-icons/fa";
import { createPortal } from "react-dom";

type MenuOwnerContextType = {
  onClose: () => void;
  x: number;
  y: number;
  x2?: number;
  y2?: number;
};

const MenuOwnerContext = createContext<MenuOwnerContextType>({ onClose: () => {}, x: 0, y: 0 });

type MenuContextType = {
  rect: DOMRect;
  activeItem: HTMLDivElement | null;
  setActiveItem: (item: HTMLDivElement | null) => void;
};

const MenuContext = createContext<MenuContextType>({
  rect: new DOMRect(),
  activeItem: null,
  setActiveItem: () => {},
});

type MenuItemProps = {
  text: string;
  disabled?: boolean;
  selected?: boolean;
  icon?: IconType;
  onClick?: () => void;
  children?: ReactNode;
};

export const MenuItem = forwardRef(
  (
    { text, disabled = false, selected = false, icon: Icon, onClick, children }: MenuItemProps,
    externalRef: Ref<HTMLDivElement | null>
  ) => {
    const { onClose } = useContext(MenuOwnerContext);
    const menu = useContext(MenuContext);
    const ref = useRef<HTMLDivElement>(null);
    const [hover, setHover] = useState(false);
    const [position, setPosition] = useState({ x: 0, x2: 0, y: 0 });

    function handleMouseEnter() {
      if (disabled) return;

      if (children) {
        menu.setActiveItem(ref.current);
      } else {
        menu.setActiveItem(null);
      }

      setHover(true);
    }

    function handleMouseLeave() {
      setHover(false);
    }

    function handleClick(event: MouseEvent<HTMLDivElement>): void {
      onClick?.();
      onClose();
      event.stopPropagation();
    }

    useLayoutEffect(() => {
      const rect = ref.current?.getBoundingClientRect();
      if (rect) {
        setPosition({ x: menu.rect.right - 8, x2: menu.rect.left + 8, y: rect.top });
      }
    }, [menu.rect, ref.current]);

    useImperativeHandle(externalRef, () => ref.current);

    const isActive = hover || menu.activeItem == ref.current || selected;
    const commonClassNames = [
      disabled ? "text-gray-500 cursor-default" : "text-white cursor-pointer",
      hover ? "bg-blue-600" : isActive ? "bg-slate-600" : "bg-zinc-800",
    ];
    const paddingClassNames = "p-1";
    const iconClassNames = [paddingClassNames, "flex items-center justify-center"];
    const textClassNames = [paddingClassNames, "select-none"];

    return (
      <>
        <div
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className={cx("col-start-1 pl-4", ...commonClassNames, ...iconClassNames)}
        >
          {Icon && <Icon />}
        </div>
        <div ref={ref} onClick={handleClick} className={cx("col-start-2", ...commonClassNames)}>
          <div className={cx(...textClassNames)} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
            {text}
          </div>
          {menu.activeItem == ref.current && (
            <MenuOwnerContext.Provider value={{ onClose: onClose, x: position.x, x2: position.x2, y: position.y }}>
              {children}
            </MenuOwnerContext.Provider>
          )}
        </div>
        <div
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className={cx("col-start-3 pr-4", ...commonClassNames, ...iconClassNames)}
        >
          {children && <FaChevronRight />}
        </div>
      </>
    );
  }
);

type MenuGroupProps = {
  label: string;
};

export const MenuGroup = ({ label }: MenuGroupProps) => {
  return (
    <div className="col-span-3 text-zinc-300 text-xs">
      <label className="p-1">{label}</label>
      <div className="h-0.5 bg-zinc-700"></div>
    </div>
  );
};

export const MenuSeparator = () => {
  return <div className="col-span-3 h-0.5 my-0.5 bg-zinc-700" />;
};

type MenuProps = {
  minWidth?: number;
  children?: ReactNode;
};

export const Menu = ({ minWidth, children }: MenuProps) => {
  const { x, x2, y, y2 } = useContext(MenuOwnerContext);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuPosition, setMenuPosition] = useState({ x, y });
  const [rect, setRect] = useState<DOMRect>(new DOMRect());
  const [activeItem, setActiveItem] = useState<HTMLDivElement | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  useLayoutEffect(() => {
    if (menuRef.current) {
      const menuWidth = menuRef.current.offsetWidth;
      const menuHeight = menuRef.current.offsetHeight;

      const newX = x + menuWidth > window.innerWidth ? (x2 ?? window.innerWidth) - menuWidth : x;
      const newY = y + menuHeight > window.innerHeight ? (y2 ?? window.innerHeight) - menuHeight : y;

      setRect(menuRef.current.getBoundingClientRect());
      if (x != newX || y != newY) {
        setMenuPosition({ x: newX, y: newY });
      }
    }
  }, [x, x2, y, menuRef.current]);

  return (
    <MenuContext.Provider value={{ rect, activeItem, setActiveItem }}>
      <div
        ref={menuRef}
        style={{
          position: "fixed",
          top: menuPosition.y,
          left: menuPosition.x,
          zIndex: 9999,
          minWidth,
          maxHeight: 300,
        }}
        className={cx(
          "min-w-max grid grid-cols-[auto,1fr,auto] bg-zinc-800 border border-zinc-600",
          isLoaded ? "transition-opacity duration-250 opacity-100" : "opacity-0",
          "overflow-y-scroll"
        )}
      >
        {children}
      </div>
    </MenuContext.Provider>
  );
};

type ContextMenuProps = {
  point: DOMPoint;
  overflowX?: number;
  overflowY?: number;
  onClose: () => void;
  children?: ReactNode;
};

export const ContextMenu = ({ point, overflowX, overflowY, onClose, children }: ContextMenuProps) => {
  function handleRightClick(event: MouseEvent<HTMLDivElement>): void {
    onClose();
    event.preventDefault();
  }

  const portalRoot = document.getElementById("context-menu-root");
  if (!portalRoot) {
    return null;
  }

  return createPortal(
    <MenuOwnerContext.Provider value={{ onClose, x: point.x, y: point.y, x2: overflowX, y2: overflowY }}>
      <div
        tabIndex={-1}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          zIndex: 9998,
        }}
        onClick={onClose}
        onBlur={onClose}
        onContextMenu={handleRightClick}
      />
      {children}
    </MenuOwnerContext.Provider>,
    portalRoot
  );
};
