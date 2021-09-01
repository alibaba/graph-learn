ROOT=`pwd`
GRPC_LIB=${ROOT}/third_party/grpc/build/lib
GFLAGS_LIB=${ROOT}/third_party/gflags/build/lib
GLOG_LIB=${ROOT}/third_party/glog/build
LD_LIBRARY_PATH=${ROOT}/built/lib:${GRPC_LIB}:${GFLAGS_LIB}:${GLOG_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
