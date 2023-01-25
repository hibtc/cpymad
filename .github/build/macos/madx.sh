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
source "$(dirname -- "${BASH_SOURCE[0]}")"/setup_compiler.sh

cd "$1"
mkdir build
cd build

if [[ ! -f CMakeCache.txt ]]; then
    pip3 install --upgrade cmake
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0077=NEW \
        -DCMAKE_POLICY_DEFAULT_CMP0042=NEW \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=OFF \
        -DCMAKE_INSTALL_PREFIX=../dist \
        -DCMAKE_BUILD_TYPE=Release \
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
fi

cmake --build . --target install -j
