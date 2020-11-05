ROOT=`pwd`
GRPC_LIB=${ROOT}/third_party/grpc/build/lib
GFLAGS_LIB=${ROOT}/third_party/gflags/build/lib
SYS_INSTALL_LIB=/usr/local/lib:/usr/local/lib64
LD_LIBRARY_PATH=${SYS_INSTALL_LIB}:${ROOT}/built/lib:${GRPC_LIB}:${GFLAGS_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
