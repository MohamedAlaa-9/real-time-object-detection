import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      // Proxy API requests (e.g., for video upload) to backend on port 8080
      '/video': 'http://localhost:8080',
      // Proxy WebSocket connections to backend on port 8080
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true
      }
    }
  }
});