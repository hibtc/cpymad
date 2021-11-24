#! /usr/bin/env bash
set -ex

for PYBIN in /opt/python/cp3*/bin; do
    "$PYBIN/pip" install -U pip
    "$PYBIN/pip" install cpymad -f dist --no-index --no-deps
    "$PYBIN/pip" install cpymad pandas pytest
    "$PYBIN/pytest" -v
done
