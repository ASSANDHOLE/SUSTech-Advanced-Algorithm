:: Note that this script is not tested on Windows Platforms, and may not work.
:: If any unexpected errors occur, please try to find a solution on the Internet.

@echo off
:: Usage: %0 <Generator>; see available generators with "cmake --help"
:: if no generator is specified, the default generator for this platform will be used
:: Example: %0 "MinGW Makefiles"
SETLOCAL
set generator=%1
cd algorithms_impl || exit /b
:: Check if the 'build' directory exists or not empty
if exist build rmdir /s /q build
set cmakeCmd=cmake -S . -B build
:: Check if the generator has been quoted
if not [%generator%]==[] (
    if not [%generator:~0,1%]==[" set generator="^%generator^%"
    set cmakeCmd=%cmakeCmd% -G %generator%
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