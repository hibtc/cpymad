#! /usr/bin/env bash
# Build cpymad from checked out sources.
#
# Usage: cpymad.sh <MADXDIR>
#
# Arguments:
#   <MADXDIR>: MAD-X installation directory
#
# Outputs:
#   ./build: builds here
#   ./dist:  places wheels here
set -ex

# Build variables:
export MADXDIR="$(readlink -nf "${1:-/mnt/dist/MAD-X}")"
export X11=0 BLAS=0 LAPACK=0
export CFLAGS="-fno-lto"
export LDFLAGS="-fno-lto"

# We create the wheels from the source distribution to verify that the
# source distribution can be used as installation medium. We will later
# upload this exact source distribution to PyPI:
python setup.py sdist

build() {
    rm -f src/cpymad/libmadx.c
    "$1/bin/pip" wheel dist/*.tar.gz --no-deps -w rawdist/
}

build /opt/python/cp310-cp310
build /opt/python/cp311-cp311
build /opt/python/cp312-cp312
build /opt/python/cp313-cp313
build /opt/python/cp313-cp313t
build /opt/python/cp314-cp314
build /opt/python/cp314-cp314t

# Bundle external shared libraries into the wheels
mkdir -p dist
for whl in rawdist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
