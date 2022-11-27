#!/bin/bash
set -e
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build || exit 1
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
cp build/libhw11_cpp.so ./
echo "Done"
