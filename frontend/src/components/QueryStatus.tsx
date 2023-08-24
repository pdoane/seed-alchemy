import { UseQueryResult } from "react-query";
import { MenuItem } from "./Menu";

type QueryStatusProps = {
  result: UseQueryResult;
};

export const QueryStatusLabel = ({ result }: QueryStatusProps) => {
  if (result.isLoading || result.isIdle) {
    return <label>Loading...</label>;
  } else if (result.isError && result.error instanceof Error) {
    return <label>{`Error: ${result.error.message}`}</label>;
  } else {
    return <label>Success</label>;
  }
};

export const QueryStatusMenuItem = ({ result }: QueryStatusProps) => {
  if (result.isLoading || result.isIdle) {
    return <MenuItem text="Loading..." disabled={true} />;
  } else if (result.isError && result.error instanceof Error) {
    return <MenuItem text={`Error: ${result.error.message}`} disabled={true} />;
  } else {
    return <MenuItem text="Success" disabled={true} />;
  }
};
