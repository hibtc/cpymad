#! /usr/bin/env bash
# Build MAD-X static library from prepared sources.
#
# Usage: madx.sh <SRCDIR>
#
# Arguments:
#   <SRCDIR>: root directory of MAD-X sources
#   <BUILDDIR>: cmake build directory
#   <INSTALLDIR>: MAD-X installation directory (binary distribution)
set -ex

SRCDIR="$(readlink -nm "${1:-/mnt/src/MAD-X}")"
BUILDDIR="$(readlink -nm "${2:-/mnt/build/MAD-X}")"
INSTALLDIR="$(readlink -nm "${3:-/mnt/dist/MAD-X}")"

mkdir -p "$BUILDDIR"
cd "$BUILDDIR"

if [[ ! -f CMakeCache.txt ]]; then
    cmake "$SRCDIR" \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=ON \
        -DCMAKE_INSTALL_PREFIX="$INSTALLDIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-fvisibility=hidden" \
        -DCMAKE_CXX_FLAGS="-fvisibility=hidden" \
        -DCMAKE_Fortran_FLAGS="-fvisibility=hidden" \
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
fi

cmake --build . --target install
