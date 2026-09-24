@echo off
REM Builds the product-side wiring check. The product's own headers are the
REM only includes it needs beyond the kernel, and the kernel's sources are
REM compiled in so the tier dispatch and the rungs are linked, not stubbed.
REM %1: the kernel checkout to wire against.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=C:\Users\myaml\dev\qcx-tier-wired
set BOYS=%~1
if "%BOYS%"=="" set BOYS=C:\Users\myaml\dev\boys-tier-wired
set OUT=%SRC%\build\wiring
if not exist "%OUT%" mkdir "%OUT%"
pushd "%OUT%"
cl /nologo /std:c++latest /O2 /DNDEBUG /EHsc ^
   /I "%SRC%\integrals\include" /I "%SRC%\integrals\src" /I "%BOYS%\include" ^
   "%SRC%\tools\auxstage\tier_wiring_check.cpp" ^
   "%BOYS%\src\boys.cpp" "%BOYS%\src\boys_simd.cpp" ^
   /Fe:tier_wiring_check.exe
echo CL_EXIT=%ERRORLEVEL%
popd
