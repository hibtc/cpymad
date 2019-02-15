:: This script requires:
:: - conda
:: - PLATFORM environment variable: either "x86" or "x64"
:: - MADXDIR environment variable: output directory from
::          python build_madx.py --static

:: Create python environments:
call conda create -qy -n py27 python=2.7 wheel
call conda create -qy -n py33 python=3.3 wheel
call conda create -qy -n py34 python=3.4 wheel
call conda create -qy -n py35 python=3.5 wheel
call conda create -qy -n py36 python=3.6 wheel
call conda create -qy -n py37 python=3.7 wheel

:: Install mingwpy where available:
call activate py27 && call pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy
call activate py33 && call pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy
call activate py34 && call pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy

:: Prepare cython source:
call activate py34
call conda install -qy cython
call cython src\cpymad\libmadx.pyx -I %MADXDIR%\include

:: Locate gcc from mingwpy in py34 (used later for build_cpymad2):
for /f %%G in ('python -c "import sys; print(sys.prefix)"') do (
    set "gcc=%%~fG\Scripts\gcc.exe"
)

:: Build cpymad wheels:
if %PLATFORM% == "x86" (
    call :build_cpymad  py27
    call :build_cpymad  py33
    call :build_cpymad  py34
    call :build_cpymad2 py35 35 win32-3.5 cp35-win32
    call :build_cpymad2 py36 36 win32-3.6 cp36-win32
    call :build_cpymad2 py37 37 win32-3.7 cp37-win32
) else (
    call :build_cpymad  py27
    call :build_cpymad  py33
    call :build_cpymad  py34
    call :build_cpymad2 py35 35 win-amd64-3.5 cp35-win_amd64
    call :build_cpymad2 py36 36 win-amd64-3.6 cp36-win_amd64
    call :build_cpymad2 py37 37 win-amd64-3.7 cp37-win_amd64
)
exit /b %ERRORLEVEL%


:: Build cpymad on py27-py34
:build_cpymad
setlocal
    set "py_env=%1"

    call activate %py_env% & @echo on
    call python setup.py build_ext -c mingw32 --static
    call python setup.py bdist_wheel
endlocal
exit /b 0


:: Build cpymad on py35+
:build_cpymad2
setlocal
    set "py_env=%1"
    set "py_ver=%2"
    set "dir_tag=%3"
    set "file_tag=%4"

    call activate %py_env% & @echo on
    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set builddir=build\lib.%dir_tag%\cpymad
    mkdir %tempdir%
    mkdir %builddir%

    for /f %%G in ('python -c "import sys; print(sys.prefix)"') do (
        set "pythondir=%%~fG"
    )

    call %gcc% -mdll -O -Wall ^
        -I%MADXDIR%\include ^
        -I%pythondir%\include ^
        -c src/cpymad/libmadx.c ^
        -o %tempdir%\libmadx.obj ^
        -std=gnu99

    :: Linking directly against the pythonXX.dll is the only way I found to
    :: satisfy the linker with conda. With WinPython, I used to successfully
    :: link using the conventional `-L%pythondir%\libs -lpython%py_ver%`
    :: command line. With conda however, the linker complains about about a
    :: large number of undefined references, such as `__imp__Py_NoneStruct`,
    :: when using this method.
    call %gcc% -shared -s ^
        %tempdir%\libmadx.obj ^
        -L%MADXDIR%\lib ^
        -lmadx -lptc -lgc-lib -lstdc++ -lgfortran ^
        -lquadmath %pythondir%\python%py_ver%.dll -lmsvcr100 ^
        -o %builddir%\libmadx.%file_tag%.pyd

    call python setup.py bdist_wheel
endlocal
exit /b 0
