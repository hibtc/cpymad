#! /usr/bin/env bash
set -ex

# Build cpymad from checked out sources.
# Expects a built madx distribution in '../MAD-X/dist'.
# Builds in './build' and places wheels in './dist'.

main()
{
    ARCH=$1
    MADXDIR=${2:-$MADXDIR}

    # Create python environments:
    conda_ create -qyf -n py27 python=2.7 wheel cython -c anaconda
    conda_ create -qyf -n py35 python=3.5 wheel cython -c anaconda
    conda_ create -qyf -n py36 python=3.6 wheel cython -c anaconda
    conda_ create -qyf -n py37 python=3.7 wheel cython -c anaconda
    conda_ create -qyf -n py38 python=3.8 wheel cython -c anaconda

    # Build cpymad wheels:
    if [[ $ARCH == i686 ]]; then
        CFLAGS=
        build py27 27 win32-2.7 ''
        build py35 35 win32-3.5 .cp35-win32
        build py36 36 win32-3.6 .cp36-win32
        build py37 37 win32-3.7 .cp37-win32
        build py38 38 win32-3.8 .cp38-win32
    else
        CFLAGS=-DMS_WIN64
        build py27 27 win-amd64-2.7 ''
        build py35 35 win-amd64-3.5 .cp35-win_amd64
        build py36 36 win-amd64-3.6 .cp36-win_amd64
        build py37 37 win-amd64-3.7 .cp37-win_amd64
        build py38 38 win-amd64-3.8 .cp38-win_amd64
    fi
}

# We manually build the C extension using our msys gcc because setuptools is
# not smart enough to figure out how to build it. The downside is that
# we link a different C runtime than is natively used by python. This will
# result in horrible evil should we ever mix C objects/memory between python
# and cpymad!
build()
{
    py_env=$1
    py_ver=$2
    dir_tag=$3
    file_tag=$4

    # Ensure that cython code and extension module will be rebuilt since the
    # cython code is partially incompatible between python versions:
    rm -f src/cpymad/libmadx.c \
          src/cpymad/libmadx.pyd

    # We use a two stage build with the exact filenames as `python setup.py
    # build_ext` would do (compile `.c` to `.obj` in $tempdir, then link to
    # `.pyd` in $libdir) to prevent the final `python setup.py bdist_wheel`
    # command from trying trying to perform either of these steps with MSVC.

    conda_ activate $py_env
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
    conda_ deactivate

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
    conda_ activate $py_env
    python setup.py bdist_wheel
    conda_ deactivate
}

conda_() {
    # Conda with disabled trace (really noisy otherwise):
    { set +x; } 2>/dev/null
    conda "$@"
    { set -x; } 2>/dev/null
}

main "$@"
