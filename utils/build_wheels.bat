:: This script builds cpymad windows wheels for all supported python versions
:: using msys in conda. It has the following dependencies:
::
:: - conda (for managing environments for different python versions)
::
:: - created environment "madx" with "m2w64-toolchain" from msys2
::
:: - %PLATFORM% environment variable set to either "x86" or "x64"
::
:: - %MADXDIR% environment variable with the output directory from
::          python build_madx.py --static
::
:: IMPORTANT: Please check the resulting `libmadx.pyd` file with "Dependency
:: Walker" (http://www.dependencywalker.com/) for non-system runtime
:: dependencies!

:: Create python environments:
call conda create -qyf -n py27 python=2.7 wheel cython -c anaconda
call conda create -qyf -n py35 python=3.5 wheel cython -c anaconda
call conda create -qyf -n py36 python=3.6 wheel cython -c anaconda
call conda create -qyf -n py37 python=3.7 wheel cython -c anaconda
call conda create -qyf -n py38 python=3.8 wheel cython -c anaconda

:: Locate gcc used during madx build (created in .appveyor.yml):
call conda activate madx & @echo on
for /f %%G in ('where gcc') do (
    set "gcc=%%~fG"
)
call conda deactivate

:: Build cpymad wheels:
if "%PLATFORM%" == "x86" (
    set "CFLAGS= "
    call :build_cpymad py27 27 win32-2.7 pyd
    call :build_cpymad py35 35 win32-3.5 cp35-win32.pyd
    call :build_cpymad py36 36 win32-3.6 cp36-win32.pyd
    call :build_cpymad py37 37 win32-3.7 cp37-win32.pyd
    call :build_cpymad py38 38 win32-3.8 cp38-win32.pyd
) else (
    set CFLAGS=-DMS_WIN64
    call :build_cpymad py27 27 win-amd64-2.7 pyd
    call :build_cpymad py35 35 win-amd64-3.5 cp35-win_amd64.pyd
    call :build_cpymad py36 36 win-amd64-3.6 cp36-win_amd64.pyd
    call :build_cpymad py37 37 win-amd64-3.7 cp37-win_amd64.pyd
    call :build_cpymad py38 38 win-amd64-3.8 cp38-win_amd64.pyd
)
exit /b %ERRORLEVEL%


:: We manually build the C extension using our msys gcc because setuptools is
:: not smart enough to figure out how to build it. The downside is that
:: we link a different C runtime than is natively used by python. This will
:: result in horrible evil should we ever mix C objects/memory between python
:: and cpymad!
:build_cpymad
    set "py_env=%1"
    set "py_ver=%2"
    set "dir_tag=%3"
    set "file_tag=%4"

    :: Ensure that cython code and extension module will be rebuilt since the
    :: cython code is partially incompatible between python versions:
    del /f src\cpymad\libmadx.c ^
           src\cpymad\libmadx.pyd

    :: We use a two stage build with the exact filenames as `python setup.py
    :: build_ext` would do (compile `.c` to `.obj` in %tempdir%, then link to
    :: `.pyd` in %libdir%) to prevent the final `python setup.py bdist_wheel`
    :: command from trying trying to perform either of these steps with MSVC.

    call conda activate %py_env% & @echo on
    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set libdir=build\lib.%dir_tag%\cpymad
    mkdir %tempdir%
    mkdir %libdir%

    for /f %%G in ('python -c "import sys; print(sys.prefix)"') do (
        set "pythondir=%%~fG"
    )

    :: This will cythonize `.pyx` to `.c`:
    call pip install -U setuptools
    call python setup.py build_py

    :: We turn back on the 'madx' environment for building in order to set the
    :: the path to the runtime DLLs required for running gcc.exe. Without this
    :: the command errors with a windows error that is visible only via the
    :: remote desktop but doesn't get logged as console output.
    call conda deactivate
    call conda activate madx & @echo on

    call %gcc% %CFLAGS% -mdll -O -Wall ^
        -I%MADXDIR%\include ^
        -I%pythondir%\include ^
        -c src/cpymad/libmadx.c ^
        -o %tempdir%\libmadx.obj ^
        -std=gnu99

    :: Linking directly against the `pythonXX.dll` is the only way I found to
    :: satisfy the linker in a conda python environment. The conventional
    :: command line `-L%pythondir%\libs -lpython%py_ver%` used to work fine on
    :: WinPython, but fails on conda with large number of complaints about
    :: about undefined references, such as `__imp__Py_NoneStruct`,
    call %gcc% -shared -s ^
        %tempdir%\libmadx.obj ^
        -L%MADXDIR%\lib ^
        -static ^
        -lmadx -lDISTlib -lptc -lgc-lib ^
        -lstdc++ -lgfortran -lquadmath ^
        %pythondir%\python%py_ver%.dll ^
        -o %libdir%\libmadx.%file_tag%

    :: Turn target python environment back on, see above:
    call conda deactivate
    call conda activate %py_env% & @echo on

    call python setup.py bdist_wheel

    call conda deactivate
exit /b 0
