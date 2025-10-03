// C:\Users\phili\meridian\apps\content-fetcher-worker\webpack.config.js
import path from 'path';
import webpack from 'webpack';
import { fileURLToPath } from 'url';

// ES Module equivalent of __filename and __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default {
  target: 'webworker',
  entry: './src/index.js',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
    libraryTarget: 'this',
    chunkFormat: 'module',
    library: {
      type: 'module',
    },
  },
  mode: 'production',
  devtool: 'cheap-module-source-map',
  resolve: {
    extensions: ['.js'],
    fallback: {
      // <<<< CRITICAL FIX: Explicitly set problematic Node.js core modules to false for nodejs_compat
      "net": false,         // Error: Can't resolve 'net'
      "tls": false,         // Error: Can't resolve 'tls'
      "perf_hooks": false,  // Error: Can't resolve 'perf_hooks'
      "os": false,          // Error: Can't resolve 'os'
      "fs": false,          // Error: Can't resolve 'fs'
      // These others were previously set to false, let's keep them explicit
      "crypto": false,
      "stream": false,
      "buffer": false,
      "events": false,
      "util": false,
      "assert": false,
      "url": false
    }
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        loader: 'babel-loader',
        options: {
          presets: ['@babel/preset-env'],
        },
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'], // This provides a Buffer global
    }),
    new webpack.DefinePlugin({
      'process.env.NODE_DEBUG': JSON.stringify(false),
      'process.versions.node': JSON.stringify(process.versions.node),
    }),
  ],
  experiments: {
    outputModule: true,
    topLevelAwait: true,
  },
  externals: [],
};