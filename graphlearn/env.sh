ROOT=`pwd`
GRPC_LIB=${ROOT}/third_party/grpc/build/lib
GFLAGS_LIB=${ROOT}/third_party/gflags/build/lib
GLOG_LIB=${ROOT}/third_party/glog/build
JAVA_LIB=${JAVA_HOME}/jre/lib/amd64/server/
LD_LIBRARY_PATH=${ROOT}/built/lib:${GRPC_LIB}:${BRANE_LIB}:${BRANE_DEPS_LIB}:${JAVA_LIB}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
