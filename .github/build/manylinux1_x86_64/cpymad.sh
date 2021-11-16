#! /usr/bin/env bash
set -ex

# Build cpymad from checked out sources.
# Expects a built madx distribution in '../MAD-X/dist'.
# Builds in './build' and places wheels in './dist'.

# Build variables:
export MADXDIR=$(readlink -nf ../MAD-X/dist)
export X11=0 BLAS=0 LAPACK=0
export CFLAGS="-fno-lto"
export LDFLAGS="-fno-lto"

# We create the wheels from the source distribution to verify that the
# source distribution can be used as installation medium. We will later
# upload this exact source distribution to PyPI:
python setup.py sdist

for PYBIN in /opt/python/cp3*/bin; do
    "${PYBIN}/pip" wheel dist/*.tar.gz --no-deps -w rawdist/
done

# Bundle external shared libraries into the wheels
mkdir -p dist
for whl in rawdist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
