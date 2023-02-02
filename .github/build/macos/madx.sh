#! /usr/bin/env bash
# Build MAD-X static library from prepared sources.
#
# Usage: madx.sh <SRCDIR> <ARCH>
#
# Arguments:
#   <SRCDIR>: root directory of MAD-X sources
#   <ARCH>:   target architecture (x86_64/arm64)
#
# Outputs:
#   <SRCDIR>/build: cmake build directory
#   <SRCDIR>/dist:  MAD-X installation directory (binary distribution)
set -ex

srcdir=$1
arch=$2

cd "$srcdir"
mkdir build
cd build

if [[ ! -f CMakeCache.txt ]]; then
    pip3 install --upgrade cmake
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0077=NEW \
        -DCMAKE_POLICY_DEFAULT_CMP0042=NEW \
        -DCMAKE_OSX_ARCHITECTURES=$arch \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=OFF \
        -DCMAKE_INSTALL_PREFIX=../dist \
        -DCMAKE_BUILD_TYPE=Release \
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
fi

cmake --build . --target install
