ROOT=`pwd`
GRPC_LIB=${ROOT}/third_party/grpc/build/lib
LD_LIBRARY_PATH=${ROOT}/built/lib:${GRPC_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
