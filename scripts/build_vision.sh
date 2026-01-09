#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vision"

rm -rf build
mkdir -p build
cd build

cmake ..
cmake --build . -j

echo "[OK] built: $ROOT/vision/bin/ght_face_eyes"
