import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom'
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // FastAPI dev
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''), // /api/chat -> /chat
      },
    },
  },
})
