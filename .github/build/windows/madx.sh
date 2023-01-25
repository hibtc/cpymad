#! /usr/bin/env bash
set -ex

# Build MAD-X static library from prepared sources.
# Must be run from the root the directory of the MAD-X sources.
# Builds in './build' and installs to './dist'.

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
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
fi

cmake --build . --target install -j
