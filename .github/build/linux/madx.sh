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

# Install glibc-static where needed and possible:
case $AUDITWHEEL_PLAT in
    manylinux1_*)
        # static package preinstalled, but under different name (?)
        ;;
    manylinux2010_*)
        yum install -y glibc-static
        ;;
    manylinux2014_*)
        yum install -y glibc-static
        ;;
    manylinux_2_24*)
        # uses apt-get, but lib seems to be unavailable here
        ;;
    manylinux_2_28*)
        yum install -y glibc-static
        ;;
    musllinux_1_1*)
        # designed for static linkage from the ground up
        ;;
esac

PATH="/opt/python/cp39-cp39/bin:$PATH"
pip install cmake --only-binary cmake

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
