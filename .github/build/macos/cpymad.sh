#! /usr/bin/env bash
set -ex
export CC=gcc-9
export MADXDIR=../MAD-X/dist
export BLAS=1 LAPACK=1

build()
{
    py_ver=$1

    conda_ create -qyf -n py$py_ver python=$py_ver wheel cython -c anaconda
    conda_ activate py$py_ver
    pip install -U setuptools
    rm -f src/cpymad/libmadx.c
    python setup.py sdist bdist_wheel
    conda_ deactivate
}

conda_() {
    # Conda with disabled trace (really noisy otherwise):
    { set +x; } 2>/dev/null
    conda "$@"
    { set -x; } 2>/dev/null
}

build 2.7
build 3.5
build 3.6
build 3.7
build 3.8
