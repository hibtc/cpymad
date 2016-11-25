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
set MADX=%~dp1
set CPYMAD=%HERE%\..

:: Cleanup remains from previous builds
cd %CPYMAD%
del /f cpymad\libmadx.pyd cpymad\libmadx.c
rd /s /q build

:: Build cpymad mingw32
call python setup.py build_ext -lquadmath -c mingw32 --madxdir=%MADX%
call python setup.py build

endlocal
