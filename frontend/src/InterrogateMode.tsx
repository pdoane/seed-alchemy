import { SessionProgress } from "./SessionProgress";

export const InterrogateMode = () => {
  return (
    <div className="flex flex-col w-full h-full">
      <SessionProgress />
      <div className="flex flex-grow overflow-hidden items-center justify-center">Interrogate Mode - Coming Soon</div>
    </div>
  );
};
