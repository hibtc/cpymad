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
mkdir build
cd build

if [[ ! -f CMakeCache.txt ]]; then
    # Build MAD-X as library:
    pip install --upgrade cmake
    cmake .. \
        -G "MinGW Makefiles" \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=ON \
        -DCMAKE_INSTALL_PREFIX=../dist \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=11 \
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
fi

cmake --build . --target install
