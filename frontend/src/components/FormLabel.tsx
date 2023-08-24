import { ReactNode } from "react";

interface FormLabelProps {
  label: string;
  children?: ReactNode;
}

export const FormLabel = ({ label, children }: FormLabelProps) => {
  return (
    <div className="w-full space-y-0.5">
      <label className="text-zinc-300">{label}</label>
      {children}
    </div>
  );
};
