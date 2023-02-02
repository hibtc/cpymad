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
set -e
export MADXDIR=$1
export BLAS=1 LAPACK=1

pip install -U setuptools wheel cython
rm -f src/cpymad/libmadx.c
python setup.py sdist bdist_wheel
