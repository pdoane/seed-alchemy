const { app, BrowserWindow } = require("electron");
const isDev = require("electron-is-dev");
const path = require("path");
const url = require("url");
const Store = require("electron-store");

const store = new Store();

function createWindow() {
  const { width, height, x, y } = store.get("windowBounds", { width: 800, height: 600 });
  const win = new BrowserWindow({
    width,
    height,
    x,
    y,
    webPreferences: {
      nodeIntegration: true,
    },
  });

  const startUrl = isDev ? "http://localhost:5173" : "http://localhost:8000";

  win.loadURL(startUrl);

  win.on("close", () => {
    const { width, height, x, y } = win.getBounds();
    store.set("windowBounds", { width, height, x, y });
  });
}

app.setAboutPanelOptions({
  applicationName: "My Electron App",
  applicationVersion: "1.0.0",
});

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
