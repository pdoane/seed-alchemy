import { stringify as uuidStringify } from "uuid";
import { useEffect } from "react";
import { stateSession } from "./store";

let ws: WebSocket | null = null;

enum MessageType {
  SESSION_ID = 1,
  PROGRESS = 2,
  IMAGE = 3,
}

function optionalUuid(bytes: Uint8Array): string | null {
  const uuid = uuidStringify(bytes);
  return uuid != "00000000-0000-0000-0000-000000000000" ? uuid : null;
}

export const WebSocketComponent = () => {
  const connect = () => {
    if (ws !== null) {
      return;
    }

    ws = new WebSocket("ws://localhost:8000/ws");
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {};

    ws.onclose = () => {
      ws = null;
      setTimeout(connect, 1000);
    };

    ws.onmessage = (event: MessageEvent) => {
      const array = new Uint8Array(event.data);
      const dataView = new DataView(event.data);

      const type = dataView.getInt32(0);
      const length = dataView.getInt32(4);

      switch (type) {
        case MessageType.SESSION_ID: {
          stateSession.sessionId = uuidStringify(array.slice(8, 24));
          break;
        }

        case MessageType.PROGRESS: {
          stateSession.generatorId = optionalUuid(array.slice(8, 24));
          const amount = dataView.getInt32(24);
          stateSession.progressAmount = amount;
          break;
        }

        case MessageType.IMAGE: {
          stateSession.generatorId = optionalUuid(array.slice(8, 24));
          const blob = new Blob([new Uint8Array(event.data, 24, length - 16)]);
          const url = URL.createObjectURL(blob);
          stateSession.previewUrl = url;
          break;
        }

        default:
          console.log("Unknown message type:", type);
          break;
      }
    };
  };

  useEffect(() => {
    connect();

    return () => {
      if (ws !== null) {
        ws.close();
        ws = null;
      }
    };
  }, []);

  return <></>;
};
