import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env': {},
    global: 'globalThis',
  },
  build: {
    rollupOptions: {
      external: []
    },
    chunkSizeWarningLimit: 1000
  },
  esbuild: {
    // Ignore TypeScript errors during build
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  }
})

