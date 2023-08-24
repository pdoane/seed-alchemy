import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000/",
        changeOrigin: true,
      },
      "/images": {
        target: "http://127.0.0.1:8000/",
        changeOrigin: true,
      },
      "/thumbnails": {
        target: "http://127.0.0.1:8000/",
        changeOrigin: true,
      },
    },
  },
});
