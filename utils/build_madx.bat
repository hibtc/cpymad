if not defined MADXDIR ( set "MADXDIR=%~dp0\madx-bin" )

:: Acquire MAD-X:
set "MADX_VER=5.05.01"
set "MADX_ZIP=%MADX_VER%.zip"
set "MADX_URL=https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/"
set "MADX_DIR=MAD-X-%MADX_VER%"
call python -m wget "%MADX_URL%/%MADX_ZIP%" -o %MADX_ZIP%
call 7za x %MADX_ZIP%

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
