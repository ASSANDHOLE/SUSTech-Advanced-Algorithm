cd algorithms_impl || exit 1
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build || exit 1
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OMP=1
make
cd ..
cp build/libalgo_cpp_impl.so ../algorithms
echo "Done"