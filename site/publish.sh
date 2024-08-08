#!/bin/sh

set -e
cd $(dirname $0)
NO_WATCH=yes ./run.sh

cp -v www/*.{wasm,js,html,css,ico} deploy/

cd deploy
git add .
git commit -m "Deploy"
git push
