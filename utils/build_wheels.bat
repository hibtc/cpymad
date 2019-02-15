:: This script requires:
:: - conda
:: - PLATFORM environment variable: either "x86" or "x64"
:: - MADXDIR environment variable: output directory from
::          python build_madx.py --static

:: Create python environments:
conda create -qy -n py27 python=2.7 wheel
conda create -qy -n py33 python=3.3 wheel
conda create -qy -n py34 python=3.4 wheel
conda create -qy -n py35 python=3.5 wheel
conda create -qy -n py36 python=3.6 wheel
conda create -qy -n py37 python=3.7 wheel

:: Install mingwpy where available:
conda install -qy -n py27 -c conda-forge mingwpy
conda install -qy -n py33 -c conda-forge mingwpy
conda install -qy -n py34 -c conda-forge mingwpy

:: Prepare cython source:
activate py34
conda install -qy cython
cython src\cpymad\libmadx.pyx -I %MADXDIR%\include

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

    activate %py_env%
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

    activate %py_env%
    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set builddir=build\lib.%dir_tag%\cpymad
    mkdir %tempdir%
    mkdir %builddir%

    for /f %%G in ('python -c "import sys; print(sys.prefix)"') do (
        set "pythondir=%%~fG"
    )

    %gcc% -mdll -O -Wall ^
        -I%MADXDIR%\include ^
        -I%pythondir%\include ^
        -c src/cpymad/libmadx.c ^
        -o %tempdir%\libmadx.obj ^
        -std=gnu99

    %gcc% -shared -s ^
        %tempdir%\libmadx.obj ^
        -L%MADXDIR%\lib ^
        -L%pythondir%\libs ^
        -lmadx -lptc -lgc-lib -lstdc++ -lgfortran ^
        -lquadmath -lpython%py_ver% -lmsvcr100 ^
        -o %builddir%\libmadx.%file_tag%.pyd

    call python setup.py bdist_wheel
endlocal
exit /b 0
