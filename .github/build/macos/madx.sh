#! /usr/bin/env bash
set -ex

# Build MAD-X static library from prepared sources.
# Must be run from the root the directory of the MAD-X sources.
# Builds in './build' and installs to './dist'.

mkdir build
cd build

if [[ ! -f CMakeCache.txt ]]; then
    pip3 install --upgrade cmake
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0077=NEW \
        -DCMAKE_POLICY_DEFAULT_CMP0042=NEW \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_C_COMPILER=gcc-9 \
        -DCMAKE_CXX_COMPILER=g++-9 \
        -DCMAKE_Fortran_COMPILER=gfortran-9 \
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
