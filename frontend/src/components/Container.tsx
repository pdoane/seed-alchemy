import { ReactNode } from "react";
import { AiOutlineDown } from "react-icons/ai";
import { cx } from "../util/classNameUtil";
import { Switch } from "./Switch";

type ContainerHeaderProps = {
  label: string;
  onClick?: () => void;
  children?: ReactNode;
};

type ContainerArrowProps = {
  value: boolean;
};

type ContainerBodyProps = {
  isOpen: boolean;
  isEnabled?: boolean;
  children?: ReactNode;
};

type CollapsibleContainerProps = {
  label: string;
  isOpen: boolean;
  onIsOpenChange: (value: boolean) => void;
  hasSwitch?: boolean;
  isEnabled?: boolean;
  onIsEnabledChange?: (value: boolean) => void;
  children?: ReactNode;
};

export const ContainerHeader = ({ label, onClick, children }: ContainerHeaderProps) => {
  return (
    <button
      className={cx(
        "flex w-full p-2 justify-between items-center",
        "rounded",
        "bg-zinc-800 hover:bg-slate-800",
        "font-bold text-left",
        "focus:outline-none"
      )}
      onClick={onClick}
    >
      {label}
      <div className="flex space-x-2 items-center">{children}</div>
    </button>
  );
};

export const ContainerArrow = ({ value }: ContainerArrowProps) => {
  return <AiOutlineDown className={cx("transform transition-transform duration-250", value ? "rotate-180" : "")} />;
};

export const ContainerBody = ({ isOpen, isEnabled = true, children }: ContainerBodyProps) => {
  return (
    <div
      className={cx("overflow-hidden transition-all duration-250", isOpen ? "max-h-full" : "max-h-0")}
      style={{ maxHeight: isOpen ? "100vh" : "0" }}
    >
      <div className={cx("w-full p-2 space-y-2 bg-zinc-700", isEnabled ? "" : "opacity-50")}>{children}</div>
    </div>
  );
};

export const CollapsibleContainer = ({
  label,
  isOpen,
  onIsOpenChange,
  hasSwitch = false,
  isEnabled = true,
  onIsEnabledChange,
  children,
}: CollapsibleContainerProps) => {
  return (
    <div className="w-full">
      <ContainerHeader label={label} onClick={() => onIsOpenChange(!isOpen)}>
        {hasSwitch && <Switch value={isEnabled} onChange={onIsEnabledChange} />}
        <ContainerArrow value={isOpen} />
      </ContainerHeader>
      <ContainerBody isOpen={isOpen} isEnabled={isEnabled}>
        {children}
      </ContainerBody>
    </div>
  );
};
