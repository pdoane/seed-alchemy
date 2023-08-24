import { ReactNode, createContext } from "react";

type RadioGroupContextType = {
  value: string;
  onChange?: (value: string) => void;
};

export const RadioGroupContext = createContext<RadioGroupContextType>({ value: "" });

type RadioGroupProps = {
  value: string;
  onChange?: (value: string) => void;
  children: ReactNode;
};

export const RadioGroup = ({ value, onChange, children }: RadioGroupProps) => {
  return <RadioGroupContext.Provider value={{ value, onChange }}>{children}</RadioGroupContext.Provider>;
};
