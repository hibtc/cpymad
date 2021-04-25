#! /usr/bin/env bash
set -ex

# Usage:
# ./cpymad.sh <MADXDIR> <python-version>

# Expects a built madx distribution in <MADXDIR>.
# Builds in './build' and places wheels in './dist'.

# We manually build the C extension using our msys gcc because setuptools is
# not smart enough to figure out how to build it. The downside is that
# we link a different C runtime than is natively used by python. This will
# result in horrible evil should we ever mix C objects/memory between python
# and cpymad!
build()
{
    MADXDIR=${1:-$MADXDIR}
    CFLAGS=-DMS_WIN64
    PLATFORM=win-amd64

    py_dot=$2
    py_ver=${py_dot/./}
    py_env=py${py_ver}
    dir_tag=${PLATFORM}-${py_dot}
    file_tag=.cp${py_ver}-${PLATFORM/-/_}

    pip install cython wheel

    # Ensure that cython code and extension module will be rebuilt since the
    # cython code is partially incompatible between python versions:
    rm -f src/cpymad/libmadx.c \
          src/cpymad/libmadx.pyd

    # We use a two stage build with the exact filenames as `python setup.py
    # build_ext` would do (compile `.c` to `.obj` in $tempdir, then link to
    # `.pyd` in $libdir) to prevent the final `python setup.py bdist_wheel`
    # command from trying trying to perform either of these steps with MSVC.

    tempdir=build/temp.$dir_tag/Release/src/cpymad
    libdir=build/lib.$dir_tag/cpymad
    mkdir -p $tempdir
    mkdir -p $libdir

    pythondir="$(python -c 'import sys; print(sys.prefix)')"

    # This will cythonize `.pyx` to `.c`:
    pip install -U setuptools
    python setup.py build_py

    # We turn back on the base environment for building in order to set the
    # the path to the runtime DLLs required for running gcc. Without this
    # the command errors with a windows error that is visible only via the
    # remote desktop but doesn't get logged as console output.

    gcc -mdll -O -Wall -flto $CFLAGS \
        -I$MADXDIR/include \
        -I$pythondir/include \
        -c src/cpymad/libmadx.c \
        -o $tempdir/libmadx.obj \
        -std=gnu99

    # Linking directly against the `pythonXX.dll` is the only way I found to
    # satisfy the linker in a conda python environment. The conventional
    # command line `-L$pythondir/libs -lpython$py_ver` used to work fine on
    # WinPython, but fails on conda with large number of complaints about
    # about undefined references, such as `__imp__Py_NoneStruct`,
    gcc -shared -s -flto \
        $tempdir/libmadx.obj \
        -L$MADXDIR/lib \
        -static \
        -lmadx -lDISTlib -lptc -lgc-lib \
        -lstdc++ -lgfortran -lquadmath \
        $pythondir/python$py_ver.dll \
        -o $libdir/libmadx$file_tag.pyd

    # Turn target python environment back on, see above:
    python setup.py bdist_wheel
}

build "$@"
