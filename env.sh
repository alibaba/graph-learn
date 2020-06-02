ROOT=`pwd`
GRPC_LIB=${ROOT}/third_party/grpc/build/lib
GFLAGS_LIB=${ROOT}/third_party/gflags/build/lib
LD_LIBRARY_PATH=${ROOT}/built/lib:${GRPC_LIB}:${GFLAGS_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
