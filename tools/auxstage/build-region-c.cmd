@echo off
REM Builds the region-C boundary probe against a boys checkout given as %1
REM (default: the product's external/boys). The kernel sources are compiled in
REM so the probe can check its reproduction against the shipped branch.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=C:\Users\myaml\dev\bladawiec
set BOYS=%~1
if "%BOYS%"=="" set BOYS=%SRC%\external\boys
set OUT=%SRC%\build\auxstage-tier
if not exist "%OUT%" mkdir "%OUT%"
pushd "%OUT%"
cl /nologo /std:c++latest /O2 /DNDEBUG /EHsc ^
   /I "%BOYS%\include" /I "%BOYS%\src" ^
   "%SRC%\tools\auxstage\region_c_ladder.cpp" ^
   "%BOYS%\src\boys.cpp" "%BOYS%\src\boys_simd.cpp" ^
   /Fe:region_c_ladder.exe
echo CL_EXIT=%ERRORLEVEL%
popd
