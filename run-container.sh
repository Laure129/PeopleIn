#!/bin/sh
set -eu

cd "$(dirname "$0")"
mkdir -p runs resources
exec podman run --rm --env-file .env \
    -v "$PWD/runs:/app/runs:Z" \
    -v "$PWD/resources:/app/resources:Z" \
    peoplein
