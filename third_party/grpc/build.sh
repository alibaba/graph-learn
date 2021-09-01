mkdir -p build
install_dir=`pwd`/build

# switch grpc to the version
cd grpc
git checkout v1.26.x
git submodule update --init

make static -j10

make install-headers prefix=${install_dir}
make install-static prefix=${install_dir}
make install-plugins prefix=${install_dir}
