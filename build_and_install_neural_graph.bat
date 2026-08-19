@echo off
setlocal

set SI_LOCATION=C:\Softimage\SOFT3D_4.0
set SDK_LOCATION=C:\Softimage\SDK_4.0
set VCVARS=C:\Program Files\Microsoft Visual Studio\VC98\Bin\VCVARS32.BAT
set SOURCE=%SDK_LOCATION%\GDK\examples\src\NeuralGraph
set DEST_BIN=%SI_LOCATION%\3D\custom\bin
set DEST_MODEL=%SI_LOCATION%\3D\custom\model
set SDK_BIN=%SDK_LOCATION%\GDK\examples\bin
set SDK_MODEL=%SDK_LOCATION%\GDK\examples\model
set LOG_DIR=C:\si3d_probe2
set LOG=%LOG_DIR%\NeuralGraph_v0_3_build.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo NeuralGraph v0.3 build started > "%LOG%"
date /t >> "%LOG%"
time /t >> "%LOG%"
echo. >> "%LOG%"

echo.
echo ============================================================
echo  Building NeuralGraph v0.3 for SOFTIMAGE^|3D 4.0
echo ============================================================
echo.
echo Close SOFTIMAGE^|3D before continuing.
echo Build log:
echo   %LOG%
echo.

if not exist "%VCVARS%" goto missing_vc
if not exist "%SDK_LOCATION%\SDKENV.BAT" goto missing_sdkenv
if not exist "%SOURCE%\NeuralGraph.cpp" goto missing_source
if not exist "%SOURCE%\NeuralGraph.cus" goto missing_cus
if not exist "%SOURCE%\mkfile.nt" goto missing_makefile
if not exist "%DEST_BIN%" goto missing_dest_bin
if not exist "%DEST_MODEL%" goto missing_dest_model

call "%VCVARS%" >> "%LOG%" 2>&1
call "%SDK_LOCATION%\SDKENV.BAT" >> "%LOG%" 2>&1

set SI_LOCATION=C:\Softimage\SOFT3D_4.0
set SDK_LOCATION=C:\Softimage\SDK_4.0
set SOURCE=%SDK_LOCATION%\GDK\examples\src\NeuralGraph
set DEST_BIN=%SI_LOCATION%\3D\custom\bin
set DEST_MODEL=%SI_LOCATION%\3D\custom\model
set SDK_BIN=%SDK_LOCATION%\GDK\examples\bin
set SDK_MODEL=%SDK_LOCATION%\GDK\examples\model

cd /d "%SOURCE%"
if errorlevel 1 goto cd_failed

echo === CLEAN === >> "%LOG%"
nmake /f mkfile.nt clobber >> "%LOG%" 2>&1
if errorlevel 1 goto clean_failed

echo === BUILD === >> "%LOG%"
nmake /f mkfile.nt >> "%LOG%" 2>&1
if errorlevel 1 goto build_failed

if not exist "%SOURCE%\NeuralGraph.dll" goto output_missing
if not exist "%SDK_BIN%" mkdir "%SDK_BIN%"
if not exist "%SDK_MODEL%" mkdir "%SDK_MODEL%"

rem Preserve the currently installed version before replacing it.
echo === BACKUP PREVIOUS INSTALLATION === >> "%LOG%"
if exist "%DEST_BIN%\NeuralGraph.dll" copy /y "%DEST_BIN%\NeuralGraph.dll" "%LOG_DIR%\NeuralGraph_previous.dll" >> "%LOG%" 2>&1
if exist "%DEST_MODEL%\NeuralGraph.cus" copy /y "%DEST_MODEL%\NeuralGraph.cus" "%LOG_DIR%\NeuralGraph_previous.cus" >> "%LOG%" 2>&1

copy /y "%SOURCE%\NeuralGraph.dll" "%SDK_BIN%\NeuralGraph.dll" >> "%LOG%" 2>&1
if errorlevel 1 goto copy_failed
copy /y "%SOURCE%\NeuralGraph.cus" "%SDK_MODEL%\NeuralGraph.cus" >> "%LOG%" 2>&1
if errorlevel 1 goto copy_failed
copy /y "%SOURCE%\NeuralGraph.dll" "%DEST_BIN%\NeuralGraph.dll" >> "%LOG%" 2>&1
if errorlevel 1 goto copy_failed
copy /y "%SOURCE%\NeuralGraph.cus" "%DEST_MODEL%\NeuralGraph.cus" >> "%LOG%" 2>&1
if errorlevel 1 goto copy_failed

if not exist "%DEST_BIN%\NeuralGraph.dll" goto verify_failed
if not exist "%DEST_MODEL%\NeuralGraph.cus" goto verify_failed

echo SUCCESS >> "%LOG%"
echo.
echo SUCCESS: NeuralGraph v0.3 was built and installed.
echo.
echo Installed:
echo   %DEST_BIN%\NeuralGraph.dll
echo   %DEST_MODEL%\NeuralGraph.cus
echo.
echo Restart SOFTIMAGE^|3D and open:
echo   Model -^> Effect -^> NeuralGraph +
echo.
goto done

:missing_vc
echo ERROR: VC6 environment script not found: %VCVARS% >> "%LOG%"
echo ERROR: VC6 environment script not found: %VCVARS%
goto failed
:missing_sdkenv
echo ERROR: SDKENV.BAT not found. >> "%LOG%"
echo ERROR: SDKENV.BAT not found: %SDK_LOCATION%\SDKENV.BAT
goto failed
:missing_source
echo ERROR: NeuralGraph.cpp not found. >> "%LOG%"
echo ERROR: NeuralGraph.cpp not found in %SOURCE%
goto failed
:missing_cus
echo ERROR: NeuralGraph.cus not found. >> "%LOG%"
echo ERROR: NeuralGraph.cus not found in %SOURCE%
goto failed
:missing_makefile
echo ERROR: mkfile.nt not found. >> "%LOG%"
echo ERROR: mkfile.nt not found in %SOURCE%
goto failed
:missing_dest_bin
echo ERROR: Softimage custom bin directory not found. >> "%LOG%"
echo ERROR: %DEST_BIN% not found.
goto failed
:missing_dest_model
echo ERROR: Softimage custom model directory not found. >> "%LOG%"
echo ERROR: %DEST_MODEL% not found.
goto failed
:cd_failed
echo ERROR: Could not enter source directory. >> "%LOG%"
echo ERROR: Could not enter %SOURCE%
goto failed
:clean_failed
echo ERROR: NMAKE clean failed. >> "%LOG%"
echo ERROR: NMAKE clean failed.
goto failed
:build_failed
echo ERROR: NMAKE build failed. >> "%LOG%"
echo ERROR: NMAKE build failed.
goto failed
:output_missing
echo ERROR: Build succeeded but NeuralGraph.dll is absent. >> "%LOG%"
echo ERROR: NeuralGraph.dll was not produced.
goto failed
:copy_failed
echo ERROR: Copy failed. Make sure SOFTIMAGE is closed. >> "%LOG%"
echo ERROR: Copy failed. Make sure SOFTIMAGE^|3D is closed.
goto failed
:verify_failed
echo ERROR: Installed-file verification failed. >> "%LOG%"
echo ERROR: Installed-file verification failed.
goto failed

:failed
echo.
echo BUILD OR INSTALL FAILED.
echo The complete log will open in Notepad:
echo   %LOG%
echo.
start notepad "%LOG%"

:done
pause
endlocal
