import React from "react";

interface JsonDisplayProps {
  data: object;
}

interface KeyValuePair {
  key: string;
  value: any;
  depth: number;
}

export const JsonDisplay = ({ data }: JsonDisplayProps) => {
  if (data === undefined) return <></>;

  const tryParseJson = (jsonString: string): object | null => {
    try {
      const obj = JSON.parse(jsonString);
      if (obj && typeof obj === "object") {
        return obj;
      }
    } catch (e) {}
    return null;
  };

  const getKeyValues = (data: any, path: string, depth: number): KeyValuePair[] => {
    const isArray = Array.isArray(data);
    const isObject = typeof data === "object" && !isArray;

    if (data === null || data === undefined) {
      return [{ key: path, value: data, depth: depth }];
    }

    if (isObject) {
      return [
        { key: path, value: "", depth: depth },
        ...Object.entries(data).flatMap(([key, value]) => {
          return getKeyValues(value, key, depth + 1);
        }),
      ];
    } else if (isArray) {
      return [
        { key: path, value: "", depth: depth },
        ...data.flatMap((value: any, index) => {
          return getKeyValues(value, index.toString(), depth + 1);
        }),
      ];
    } else if (typeof data === "string") {
      const parsedData = tryParseJson(data);
      if (parsedData) {
        return getKeyValues(parsedData, path, depth);
      } else {
        return [{ key: path, value: data, depth: depth }];
      }
    }
    return [{ key: path, value: JSON.stringify(data), depth: depth }];
  };

  return (
    <>
      {Object.entries(data).map(([key, value], index) => {
        const keyValuePairs = getKeyValues(value, "", 0);
        return (
          <div key={index}>
            <span className="font-bold text-xl text-zinc-500">{key}</span>
            <div className="grid gap-y-1" style={{ gridTemplateColumns: "auto 1fr" }}>
              {keyValuePairs.map((pair, pairIndex) => (
                <React.Fragment key={pairIndex}>
                  <div
                    className="px-2 font-bold text-left text-zinc-400"
                    style={{ paddingLeft: `${pair.depth * 10}px` }}
                  >
                    {pair.key}
                  </div>
                  <div className="overflow-hidden text-ellipsis">{pair.value}</div>
                </React.Fragment>
              ))}
            </div>
          </div>
        );
      })}
    </>
  );
};
