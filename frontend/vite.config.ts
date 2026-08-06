import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/**
 * The build lands in `static/` at the repo root, which FastAPI mounts. One
 * container serves both the API and the bundle, which is what a Hugging Face
 * Space gives you -- and it removes CORS entirely.
 */
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    // A demo that a recruiter opens cold is already waiting on a 40s backend
    // wake-up; the bundle should not add to it.
    chunkSizeWarningLimit: 300,
  },
  server: {
    port: 5173,
    // Only used by `npm run dev`. In production the API is same-origin, so
    // there is no proxy and no CORS configuration to keep in sync.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
