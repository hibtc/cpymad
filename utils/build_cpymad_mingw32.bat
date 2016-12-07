:: Only invoke this script with a reasonable MinGW build environment (à la
:: WinPython) activated.
::
:: Parameters:
::      %1      MAD-X install prefix
::
@setlocal
@echo off

:: Get command parameters
set HERE=%~dp0
set MADX=%~f1
set CPYMAD=%HERE%\..

if not exist %MADX%\CMakeLists.txt goto usage

:: Cleanup remains from previous builds
cd %CPYMAD%
del /f cpymad\libmadx.pyd cpymad\libmadx.c
rd /s /q build

:: Build cpymad mingw32
call python setup.py build_ext -lquadmath -c mingw32 --madxdir=%MADX%
call python setup.py build

goto end

:usage
echo Usage: .\build_cpymad_mingw32.bat MADX_PATH
pause

:end
endlocal
