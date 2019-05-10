if not defined MADXDIR ( set "MADXDIR=%~dp0\madx-bin" )

:: Acquire MAD-X:
set "MADX_VER=5.05.00"
set "MADX_ZIP=%MADX_VER%.zip"
set "MADX_URL=https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/"
set "MADX_DIR=MAD-X-%MADX_VER%"
call python -m wget "%MADX_URL%/%MADX_ZIP%" -o %MADX_ZIP%
call 7za x %MADX_ZIP%

:: Patch garbage collector to newer version, see #41:
set "GC_VER=8.0.2"
set "GC_ZIP=gc-%GC_VER%.tar.gz"
set "GC_URL=https://github.com/ivmai/bdwgc/releases/download/"
call python -m wget "%GC_URL%/v%GC_VER%/%GC_ZIP%" -o %GC_ZIP%
call 7za x %GC_ZIP%
call 7za x -o%MADX_DIR%\libs\gc gc-%GC_VER%.tar
call patch -d %MADX_DIR% -p1 -i "%~dp0\patches\gc-8.0.2.diff"

:: Build MAD-X as library:
mkdir "%MADX_DIR%\build"
cd "%MADX_DIR%\build"
call cmake .. -G "MinGW Makefiles" ^
    -DMADX_ONLINE=OFF ^
    -DMADX_INSTALL_DOC=OFF ^
    -DCMAKE_INSTALL_PREFIX=%MADXDIR% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DMADX_STATIC=ON ^
    -DBUILD_SHARED_LIBS=OFF

call mingw32-make install
cd ..\..
