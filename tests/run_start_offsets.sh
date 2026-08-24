#!/bin/sh
set -eu

for offset in 40 80 120 160; do
    uv run --python 3.12 --with-requirements requirements.txt \
        python -m peoplein.run --start-offset-ms "$offset"
done
