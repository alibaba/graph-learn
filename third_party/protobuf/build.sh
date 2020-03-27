mkdir -p build
prefix=`pwd`/build

cd protobuf/
git checkout 3.10.x
git submodule update --init --recursive
./autogen.sh
./configure --prefix=${prefix} --disable-shared CXXFLAGS=-fPIC
make -j10
make install
