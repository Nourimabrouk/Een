import { defineConfig } from 'vite';
import { resolve } from 'path';
import legacy from '@vitejs/plugin-legacy';
import viteCompression from 'vite-plugin-compression';

export default defineConfig({
  // Base directory for serving files
  root: 'website',
  
  // Public directory (for static assets)
  publicDir: '../public',
  
  // Build configuration
  build: {
    // Output directory (relative to project root)
    outDir: '../dist',
    
    // Empty the output directory before build
    emptyOutDir: true,
    
    // Assets directory name
    assetsDir: 'assets',
    
    // Enable source maps for debugging
    sourcemap: true,
    
    // Rollup options for optimization
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'website/metastation-hub.html'),
        zen: resolve(__dirname, 'website/zen-unity-meditation.html'),
        consciousness: resolve(__dirname, 'website/consciousness_dashboard.html'),
        framework: resolve(__dirname, 'website/mathematical-framework.html'),
        implementations: resolve(__dirname, 'website/implementations-gallery.html'),
        // Add more entry points as needed
      },
      output: {
        // Manual chunk splitting for better caching
        manualChunks: {
          'three': ['three'],
          'gsap': ['gsap']
        }
      }
    },
    
    // Chunk size warnings
    chunkSizeWarningLimit: 1000,
    
    // Minification options
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  
  // Development server configuration
  server: {
    port: 8001,
    host: 'localhost',
    open: '/metastation-hub.html',
    
    // Enable CORS for external resources
    cors: true,
    
    // HMR (Hot Module Replacement) configuration
    hmr: {
      overlay: true
    }
  },
  
  // Preview server (production build preview)
  preview: {
    port: 8002,
    host: 'localhost'
  },
  
  // Plugins
  plugins: [
    // Legacy browser support
    legacy({
      targets: ['defaults', 'not IE 11']
    }),
    
    // Gzip compression
    viteCompression({
      algorithm: 'gzip',
      ext: '.gz'
    }),
    
    // Brotli compression
    viteCompression({
      algorithm: 'brotliCompress',
      ext: '.br'
    })
  ],
  
  // Dependency optimization
  optimizeDeps: {
    include: ['three', 'gsap'],
    exclude: []
  },
  
  // CSS configuration
  css: {
    // CSS modules
    modules: {
      localsConvention: 'camelCase'
    },
    
    // PostCSS configuration
    postcss: {
      plugins: []
    },
    
    // Preprocessor options
    preprocessorOptions: {
      scss: {
        additionalData: `@import "./website/css/variables.scss";`
      }
    }
  },
  
  // Resolve configuration
  resolve: {
    alias: {
      '@': resolve(__dirname, 'website'),
      '@js': resolve(__dirname, 'website/js'),
      '@css': resolve(__dirname, 'website/css'),
      '@assets': resolve(__dirname, 'website/assets')
    }
  },
  
  // Environment variables
  define: {
    '__APP_VERSION__': JSON.stringify(process.env.npm_package_version),
    '__BUILD_DATE__': JSON.stringify(new Date().toISOString())
  }
});