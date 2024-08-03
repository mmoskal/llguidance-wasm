#!/bin/sh

cd $(dirname $0)

set -e
test -d node_modules || yarn install
cd node_modules
rm -f llguidance-wasm node-llama-cpp
ln -s ../../pkg llguidance-wasm
ln -s ../../../node-llama-cpp .
cd ..

echo "Building..."
./node_modules/.bin/tsc 
echo "Running..."
node dist/index.js
