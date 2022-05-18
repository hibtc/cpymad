#! /usr/bin/env bash
set -ex
export CC=gcc-9
export MADXDIR=../MAD-X/dist
export BLAS=1 LAPACK=1

build()
{
    py_ver=$1

    _ conda create -qyf -n py$py_ver python=$py_ver -c anaconda
    _ conda activate py$py_ver
    pip install -U setuptools wheel cython
    rm -f src/cpymad/libmadx.c
    python setup.py sdist bdist_wheel
    _ conda deactivate
}

_() {
    # run command with disabled trace to decrease noise
    { set +x; } 2>/dev/null
    "$@"; exitcode=$?
    { set -x; return $exitcode; } 2>/dev/null
}

build 3.6
build 3.7
build 3.8
build 3.9
build 3.10
