#! /usr/bin/env bash
# Build cpymad from checked out sources.
#
# Usage: cpymad.sh <MADXDIR> <PYVER>
#
# Arguments:
#   <MADXDIR>: MAD-X installation directory
#   <PYVER>:   python version (e.g. "3.9")
#
# Outputs:
#   ./build: builds here
#   ./dist:  places wheels here
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")"/setup_compiler.sh
export MADXDIR=$1
export PYVER=$2
export BLAS=1 LAPACK=1

conda create -qyf -n py$PYVER python=$PYVER -c anaconda
conda activate py$PYVER
pip install -U setuptools wheel cython
rm -f src/cpymad/libmadx.c
python setup.py sdist bdist_wheel
conda deactivate
