import {
  Children,
  Fragment,
  MouseEvent,
  ReactNode,
  createContext,
  isValidElement,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { AiOutlineDown } from "react-icons/ai";
import { cx } from "../util/classNameUtil";
import { ContextMenu, Menu, MenuItem } from "./Menu";

type SelectProps = {
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
  children?: ReactNode;
  placement?: string;
  status?: string;
};

type SelectItemProps = {
  value: string;
  text: string;
};

type SelectOwnerContextType = {
  value?: string;
  onChange?: (value: string) => void;
};

const SelectOwnerContext = createContext<SelectOwnerContextType>({ onChange: undefined, value: "" });

export const SelectItem = ({ value, text }: SelectItemProps) => {
  const owner = useContext(SelectOwnerContext);
  const itemRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (value == owner.value) {
      itemRef.current?.scrollIntoView({ block: "center" });
    }
  }, [value, owner.value]);

  function handleClick(): void {
    owner.onChange?.(value);
  }

  return <MenuItem ref={itemRef} text={text} selected={value == owner.value} onClick={handleClick} />;
};

const findSelectedText = (children: ReactNode, value: string): string | null => {
  let result: string | null = null;
  Children.forEach(children, (child) => {
    if (isValidElement(child)) {
      if (child.type === Fragment) {
        result = findSelectedText(child.props.children, value) || result;
      } else if (child.props.value === value) {
        result = child.props.text;
      }
    }
  });
  return result;
};

export const Select = ({ value, placeholder, onChange, children, status }: SelectProps) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [contextMenuPoint, setContextMenuPoint] = useState<DOMPoint | null>(null);
  const [overflowY, setOverflowY] = useState<number>(0);
  const [menuWidth, setMenuWidth] = useState<number>(100);
  const [currentText, setCurrentText] = useState<string | null>(null);

  useEffect(() => {
    let newText = null;
    if (value !== undefined) {
      newText = findSelectedText(children, value);
    }
    setCurrentText(newText ?? placeholder ?? "");
  }, [value, children]);

  function handleClick(event: MouseEvent<HTMLButtonElement>): void {
    const rect = event.currentTarget.getBoundingClientRect();
    setContextMenuPoint(new DOMPoint(rect.left, rect.bottom + 2));
    setOverflowY(rect.top - 2);
  }

  useLayoutEffect(() => {
    if (buttonRef.current) {
      setMenuWidth(buttonRef.current.offsetWidth);
    }
  }, [buttonRef.current]);

  return (
    <div className="w-full">
      <button
        ref={buttonRef}
        className={cx(
          "w-full p-1 flex items-center justify-between",
          "bg-zinc-950",
          "focus:ring-2 focus:ring-blue-500",
          "hover:ring-2 hover:ring-slate-500",
          "transition duration-250 ease-in-out"
        )}
        onClick={handleClick}
      >
        <label>
          {status === "loading" ? "Loading..." : status === "error" ? "Error" : currentText ?? placeholder}
        </label>
        <AiOutlineDown />
      </button>
      {contextMenuPoint && (
        <SelectOwnerContext.Provider value={{ onChange, value: value }}>
          <ContextMenu point={contextMenuPoint} overflowY={overflowY} onClose={() => setContextMenuPoint(null)}>
            <Menu minWidth={menuWidth}>{children}</Menu>
          </ContextMenu>
        </SelectOwnerContext.Provider>
      )}
    </div>
  );
};
