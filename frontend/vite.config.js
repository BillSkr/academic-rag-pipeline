import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/query': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/build': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true
      }
    }
  }
})
