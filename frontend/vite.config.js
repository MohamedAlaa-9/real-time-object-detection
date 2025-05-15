import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      // Proxy API requests to backend
      '/api': {
        target: 'https://adelkazzaz.ninja',
        changeOrigin: true,
        secure: true
      },
      // Proxy WebSocket connections to backend
      '/ws': {
        target: 'wss://adelkazzaz.ninja',
        ws: true,
        secure: true,
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false
  }
});