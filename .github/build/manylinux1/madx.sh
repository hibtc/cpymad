#! /usr/bin/env bash
set -ex

# Build MAD-X static library from prepared sources.
# Must be run from the root the directory of the MAD-X sources.
# Builds in './build' and installs to './dist'.

rm -rf build
mkdir build
cd build

: ${PY:=/opt/python/cp36-cp36m/bin}

$PY/pip install --upgrade cmake
$PY/cmake .. \
    -DBUILD_SHARED_LIBS=OFF \
    -DMADX_STATIC=ON \
    -DCMAKE_INSTALL_PREFIX=../dist \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fvisibility=hidden -flto" \
    -DCMAKE_CXX_FLAGS="-fvisibility=hidden -flto" \
    -DCMAKE_Fortran_FLAGS="-fvisibility=hidden -flto" \
    -DMADX_INSTALL_DOC=OFF \
    -DMADX_ONLINE=OFF \
    -DMADX_FORCE_32=OFF \
    -DMADX_X11=OFF
$PY/cmake --build . --target install
