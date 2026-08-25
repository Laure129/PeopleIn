#!/bin/sh
set -eu

cd "$(dirname "$0")"
mkdir -p runs resources
if [ -f .env ]; then
    set -- --env-file .env
else
    set --
fi
exec podman run --rm "$@" \
    -v "$PWD/runs:/app/runs:Z" \
    -v "$PWD/resources:/app/resources:Z" \
    peoplein
