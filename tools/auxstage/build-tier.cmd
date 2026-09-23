@echo off
REM Builds the run-time tier validator against a boys checkout given as %1
REM (default: the product's external/boys). The kernel sources are compiled in,
REM the way integrals/CMakeLists.txt consumes them, so the m = 1 symbols the
REM extern-template declarations route out of the header are present.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=C:\Users\myaml\dev\bladawiec
set BOYS=%~1
if "%BOYS%"=="" set BOYS=%SRC%\external\boys
set OUT=%SRC%\build\auxstage-tier
if not exist "%OUT%" mkdir "%OUT%"
pushd "%OUT%"
cl /nologo /std:c++latest /O2 /DNDEBUG /EHsc ^
   /I "%BOYS%\include" /I "%BOYS%\src" ^
   "%SRC%\tools\auxstage\tier_api_check.cpp" ^
   "%BOYS%\src\boys.cpp" "%BOYS%\src\boys_simd.cpp" ^
   /Fe:tier_api_check.exe
echo CL_EXIT=%ERRORLEVEL%
popd
