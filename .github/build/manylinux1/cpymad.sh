#! /usr/bin/env bash
set -ex

# Build cpymad from checked out sources.
# Expects a built madx distribution in '../MAD-X/dist'.
# Builds in './build' and places wheels in './dist'.

: ${PY:=/opt/python/cp36-cp36m/bin}

# Build variables:
export MADXDIR=$(readlink -nf ../MAD-X/dist)
export X11=0 BLAS=0 LAPACK=0
export CFLAGS="-fno-lto"
export LDFLAGS="-fno-lto"

# Copy the cpymad source files to a build folder in order to avoid permission
# issues with the host filesystem (on both sides):
mkdir -p build
$PY/python setup.py egg_info
tar -c $(cat src/cpymad.egg-info/SOURCES.txt) |
    tar -x -C build --no-same-owner

# We create the wheels from the source distribution to verify that the
# source distribution can be used as installation medium. We will later
# upload this exact source distribution to PyPI:
pushd build
$PY/pip install cython
$PY/python setup.py sdist
$PY/pip uninstall cython -y

for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -U setuptools
    "${PYBIN}/pip" wheel dist/*.tar.gz --no-deps -w dist/
done
popd

# Bundle external shared libraries into the wheels
mkdir -p dist
for whl in build/dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
cp build/dist/*.tar.gz dist/
