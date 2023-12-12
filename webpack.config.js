// webpack.config.js
const path = require('path');
const webpack = require('webpack');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const CopyPlugin = require("copy-webpack-plugin");


module.exports = {
    mode: 'production',
    devtool: false,
    entry: './index.js',
    target: ['web'],
    output: {
        path: path.resolve(__dirname, './dist'),
        filename: 'facerecognition-sdk.js',
        // globalObject: 'this',
        library: {
            type: 'umd'
        }
    },

    plugins: [
        new CleanWebpackPlugin(),
        // new webpack.SourceMapDevToolPlugin({
        //     filename: 'facerecognition-sdk.js.map'
        // }),
        new CopyPlugin({
            // Use copy plugin to copy *.wasm to output folder.
            patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
        })
    ],
    module: {
        rules: [
          {
            test: /\.(js)$/,
            exclude: /node_modules/,
            use: "babel-loader",
          },
        ],
  },
}
