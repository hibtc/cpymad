:: Only invoke this script with a reasonable MinGW build environment (à la
:: WinPython) activated.
::
:: Parameters:
::      %1      MAD-X source path
::      %2      build directory
::      %3      install prefix
::
@setlocal
@echo off

:: Get command parameters
set HERE=%~dp0
set SRC=%~f1
set BUILD=%~f2
set PREFIX=%~f3
set UTILS=%HERE%

if not exist %SRC% goto usage

:: Remove files starting with "._" (in corrupt MAD-X tarballs)
cd %SRC%
for /F %%f in ('dir /A:-D /b /s ._*') do del %%f

:: Run CMake in build directory
call :cleandir %BUILD%
cd %BUILD%
cmake -G "MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=%PREFIX% -DBUILD_SHARED_LIBS=OFF -DMADX_NTPSA=OFF ..

:: fix "-gcc_eh" option
python %UTILS%\_replace.py "-lgcc_eh" "" -i %BUILD%\src\CMakeFiles\madxbin.dir\link.txt
python %UTILS%\_replace.py "-lgcc_eh" "" -i %BUILD%\src\CMakeFiles\madxbin.dir\linklibs.rsp

:: build MAD-X
mingw32-make

::
call :cleandir %PREFIX%
mingw32-make install

goto end


:: ------------------
:: Internal functions
:: ------------------

:cleandir
if exist %1 (
 set /p confirm="Delete files in %1? YES/NO: "
 if NOT "%confirm%" == "YES" (
  goto end
 )
 rd /s /q %1
 mkdir %1
)
exit /b 0

:usage
echo Usage: .\build_madx_mingw32.bat MADX_PATH BUILD_PATH INSTALL_PATH
pause
goto end

:end
endlocal
