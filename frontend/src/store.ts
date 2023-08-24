import { proxy, subscribe } from "valtio";
import { subscribeKey } from "valtio/utils";
import { Loadable, loadNew } from "./util/loadUtil";
import { CanvasModeState, SessionState, SettingsState, SystemState } from "./schema";

export function loadFromLocalStorage<T extends Loadable>(type: new () => T, key: string): T {
  return loadNew(JSON.parse(localStorage.getItem(key) || "{}"), type);
}

export const stateCanvas = proxy(loadFromLocalStorage(CanvasModeState, "seed-alchemy-canvas"));
export const stateSession = proxy(new SessionState());
export const stateSystem = proxy(loadFromLocalStorage(SystemState, "seed-alchemy-system"));
export const stateSettings = proxy(new SettingsState());

// Persistence
// - session is not persisted
// - system uses local storage
// - settings uses per-user server storage

// Deep assignment to preserve proxy subscriptions
function assign(dst: any, src: any) {
  if (Array.isArray(dst)) {
    for (let i = 0; i < src.length; i++) {
      if (i < dst.length) {
        if (typeof dst[i] === "object") assign(dst[i], src[i]);
        else dst[i] = src[i];
      } else {
        dst.push(src[i]);
      }
    }
    dst.length = src.length;
  } else {
    for (const key in dst) {
      if (typeof dst[key] === "object") assign(dst[key], src[key]);
      else dst[key] = src[key];
    }
  }
}

// Get latest data from server
async function fetchSettings() {
  const user = stateSystem.user;
  const response = await fetch(`/api/v1/settings/${user}`);
  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  const newSettings = loadNew(await response.json(), SettingsState);
  assign(stateSettings, newSettings);
}

// Persists to local storage
subscribe(stateSystem, async () => {
  localStorage.setItem("seed-alchemy-system", JSON.stringify(stateSystem));
});

subscribe(stateCanvas, async () => {
  localStorage.setItem("seed-alchemy-canvas", JSON.stringify(stateCanvas));
});

// Subscribe to user changes
subscribeKey(stateSystem, "user", fetchSettings);

// Get initial data from server
fetchSettings();
