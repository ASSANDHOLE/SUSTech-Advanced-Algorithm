@echo off
:: Usage: %0 <Generator>; see available generators with "cmake --help"
:: if no generator is specified, the default generator for this platform will be used
:: Example: %0 "MinGW Makefiles"
SETLOCAL
set generator=%1
set genFirstChar=%generator:~0,1%
set genLastChar=%generator:~-1%
set genFirstChar=%genFirstChar:"=+%
set genLastChar=%genLastChar:"=+%
set isQuoted=0
if "%genFirstChar%"=="+" if "%genLastChar%"=="+" (
     set isQuoted=1
)

cd algorithms_impl || exit /b
:: Check if the 'build' directory exists or not empty
if exist build rmdir /s /q build || exit /b
set "cmakeCmd=cmake -S . -B build"
:: Check if the generator has been quoted
if not [%generator%]==[] (
    if %isQuoted%==0 set generator="%generator%"
    set "cmakeCmd=%cmakeCmd% -G %generator%"
)
set cmakeCmd=%cmakeCmd% -DCMAKE_BUILD_TYPE=Release
:: Run CMake
echo Building algorithms_impl... with %cmakeCmd%
%cmakeCmd% || exit /b
:: Build
cmake --build build || exit /b
cd .. || exit /b
:: Copy the 'libalgo_cpp_impl.dll' to the 'algorithms' directory
copy algorithms_impl\build\libalgo_cpp_impl.dll algorithms\libalgo_cpp_impl.dll || exit /b
echo Done.
ENDLOCAL