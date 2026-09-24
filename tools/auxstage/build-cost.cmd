@echo off
REM Builds the runtime-tier dispatch cost probe. The kernel's own sources are
REM compiled in, the way integrals/CMakeLists.txt consumes them, so the m = 1
REM symbols the extern-template declarations route out of the header are
REM present without linking the product tree.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=C:\Users\myaml\dev\bladawiec
set BOYS=%SRC%\external\boys
set OUT=%SRC%\build\auxstage-harness
if not exist "%OUT%" mkdir "%OUT%"
cl /nologo /std:c++latest /O2 /DNDEBUG /EHsc ^
   /I "%BOYS%\include" /I "%BOYS%\src" ^
   /Fo"%OUT%\\" /Fe"%OUT%\tier_dispatch_cost.exe" ^
   "%SRC%\tools\auxstage\tier_dispatch_cost.cpp" ^
   "%BOYS%\src\boys.cpp" "%BOYS%\src\boys_simd.cpp"
echo CL_EXIT=%ERRORLEVEL%
