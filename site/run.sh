#!/bin/sh

cd $(dirname $0)

set -e
test -d node_modules || yarn install
cd node_modules
rm -f llguidance-wasm node-llama-cpp guidance-ts
ln -s ../../pkg llguidance-wasm
ln -s ../../../guidance-ts .
cd ..

cd ..
wasm-pack build --target web --no-opt --release
cd site

echo "Building..."

ESBUILD="./node_modules/.bin/esbuild
    ./ts/index.ts
    --bundle --outfile=./www/bundle.js --format=esm --sourcemap"

$ESBUILD

$ESBUILD \
    --serve=8042 \
    --servedir=www
