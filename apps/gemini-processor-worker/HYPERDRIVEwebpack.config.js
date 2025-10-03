// C:\Users\phili\meridian\apps\gemini-processor-worker\webpack.config.js
import path from 'path';
import webpack from 'webpack';
import { fileURLToPath } from 'url';

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
      "fs": false,
      "net": false,
      "tls": false,
      "crypto": false,
      "stream": false,
      "buffer": false,
      "events": false,
      "util": false,
      "assert": false,
      "url": false,
      "os": false,
      "perf_hooks": false,
      "async_hooks": false,
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
      Buffer: ['buffer', 'Buffer'],
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