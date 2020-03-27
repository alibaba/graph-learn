mkdir -p build
cd build
cmake ../glog
make
cp ../glog/src/glog/log_severity.h ./glog/
