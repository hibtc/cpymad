#! /usr/bin/env bash
export CC=gcc-9
export MADXDIR=../MAD-X/dist
export BLAS=1 LAPACK=1
pip install cython wheel
python setup.py sdist bdist_wheel -p macosx_10_9_x86_64
