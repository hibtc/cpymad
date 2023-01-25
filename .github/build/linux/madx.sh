#! /usr/bin/env bash
# Build MAD-X static library from prepared sources.
#
# Usage: madx.sh <SRCDIR>
#
# Arguments:
#   <SRCDIR>: root directory of MAD-X sources
#
# Outputs:
#   <SRCDIR>/build: cmake build directory
#   <SRCDIR>/dist:  MAD-X installation directory (binary distribution)
set -ex

cd "$1"
mkdir -p build
cd build

if [[ ! -f CMakeCache.txt ]]; then
    cmake .. \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=ON \
        -DCMAKE_INSTALL_PREFIX=../dist \
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
