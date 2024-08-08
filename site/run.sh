#!/bin/sh

cd $(dirname $0)

set -e
test -d node_modules || yarn install

cd node_modules
rm -f llguidance-wasm guidance-ts window-ai-ll-polyfill
ln -s ../../pkg llguidance-wasm
ln -s ../../../guidance-ts guidance-ts
ln -s ../../../web-llm window-ai-ll-polyfill
cd ..

(cd ../../guidance-ts && yarn run build)
(cd .. && wasm-pack build --target web --no-opt --release)
(cd www && rm -f *.wasm && ln -s ../../pkg/llguidance_wasm_bg.wasm .)

echo "Building..."

ESBUILD="./node_modules/.bin/esbuild
    ./ts/index.ts
    --bundle --outfile=./www/bundle.js --format=esm --sourcemap"

$ESBUILD

if [ "X$NO_WATCH" = "X" ]; then
$ESBUILD \
    --serve=8042 \
    --servedir=www
fi
