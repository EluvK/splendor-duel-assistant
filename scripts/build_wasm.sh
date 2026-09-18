#!/usr/bin/env bash
set -e

echo "==> Building splendor-duel-wasm target..."
cargo build --package splendor-duel-wasm --target wasm32-unknown-unknown --release

echo "==> Running wasm-bindgen..."
wasm-bindgen --target web --out-dir engine/web/pkg target/wasm32-unknown-unknown/release/splendor_duel_wasm.wasm

echo "==> Syncing best.onnx model and metadata..."
mkdir -p engine/web/assets/models
if [ -f checkpoints/best.onnx ]; then
  cp checkpoints/best.onnx engine/web/assets/models/best.onnx
fi
if [ -f checkpoints/best.json ]; then
  cp checkpoints/best.json engine/web/assets/models/best.json
fi

echo "==> Done! Output generated in engine/web/pkg"
