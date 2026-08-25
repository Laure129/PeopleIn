#!/bin/sh
set -eu

cd "$(dirname "$0")"
mkdir -p runs resources
echo "Building peoplein image..."
podman build -t peoplein .
if [ -f .env ]; then
    set -- --env-file .env
else
    set --
fi
echo "Starting peoplein..."
exec podman run --rm --log-driver=passthrough "$@" \
    -v "$PWD/runs:/app/runs:Z" \
    -v "$PWD/resources:/app/resources:Z" \
    peoplein
