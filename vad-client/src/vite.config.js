// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// Import the polyfill plugin
import { NodeGlobalsPolyfillPlugin } from '@esbuild-plugins/node-globals-polyfill';

export default defineConfig({
  plugins: [react()],
  // Remove the simple define for process.env and global
  // define: {
  //   'process.env': process.env,
  //    global: 'window',
  // },

  // Configure esbuild for dependency optimization during dev
  optimizeDeps: {
    esbuildOptions: {
      // Node.js global to ESM
      define: {
        global: 'globalThis', // Use globalThis for better compatibility
      },
      // Enable esbuild polyfill plugins
      plugins: [
        NodeGlobalsPolyfillPlugin({
          process: true, // Polyfill process
          buffer: true,  // Polyfill Buffer (often needed with process)
        }),
      ],
    },
  },
  // Configure rollup for the production build
  build: {
    rollupOptions: {
      plugins: [
        // Enable rollup polyfills plugin
        NodeGlobalsPolyfillPlugin({
          process: true,
          buffer: true,
        }),
      ],
    },
  },
});
