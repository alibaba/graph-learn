# switch to the version
cd googletest
git checkout v1.8.x
cd ..

mkdir -p build
cd build
cmake ../googletest
make
