@echo off
REM Builds the ladder probe. The kernel's own two sources are compiled in, the
REM way integrals/CMakeLists.txt consumes them, so the m = 1 symbols the
REM extern-template declarations route out of the header are present without
REM linking the product tree.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=C:\Users\myaml\dev\bladawiec
set BOYS=%SRC%\external\boys
set OUT=%SRC%\build\auxstage-harness
if not exist "%OUT%" mkdir "%OUT%"
cl /nologo /std:c++latest /O2 /DNDEBUG /EHsc ^
   /I "%BOYS%\include" /I "%BOYS%\src" ^
   /Fo"%OUT%\\" /Fe"%OUT%\m_ladder.exe" ^
   "%SRC%\tools\auxstage\m_ladder.cpp" ^
   "%BOYS%\src\boys.cpp" "%BOYS%\src\boys_simd.cpp"
echo CL_EXIT=%ERRORLEVEL%
