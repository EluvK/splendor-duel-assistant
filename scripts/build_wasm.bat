@echo off
echo ==> Building splendor-duel-wasm target...
cargo build --package splendor-duel-wasm --target wasm32-unknown-unknown --release
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==> Running wasm-bindgen...
wasm-bindgen --target web --out-dir engine/web/pkg target/wasm32-unknown-unknown/release/splendor_duel_wasm.wasm
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==> Syncing best.onnx model...
if not exist "engine\web\assets\models" mkdir "engine\web\assets\models"
if exist "checkpoints\best.onnx" copy /y "checkpoints\best.onnx" "engine\web\assets\models\best.onnx"

echo ==> Done! Output generated in engine\web\pkg
