HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := ../../..
BUILT_DIR := $(ROOT)/built
LIB_DIR := $(BUILT_DIR)/lib
BIN_DIR := $(BUILT_DIR)/bin
THIRD_PARTY_DIR := $(ROOT)/third_party

GLOG_DIR := $(THIRD_PARTY_DIR)/glog
GLOG_INCLUDE := $(GLOG_DIR)/build

PROTOBUF_DIR := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_INCLUDE := $(PROTOBUF_DIR)/build/include

GTEST_DIR := $(THIRD_PARTY_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/googletest/googletest/include
GTEST_LIB := $(GTEST_DIR)/build/googlemock/gtest

FAISS_PKG := faiss_gcc4.8.5.tar.gz
FAISS_RESOURCE := https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/$(FAISS_PKG)
FAISS_DIR := $(HERE)/faiss
FAISS_INCLUDE := $(FAISS_DIR)/include
FAISS_LIB := $(FAISS_DIR)/lib

GPU := CLOSE
CXX := g++
CXXFLAGS := $(MODEFLAGS) -std=c++11 -fPIC          \
            -pthread -mavx -msse4.2 -msse4.1       \
            -D$(GPU)_GPU                           \
            -I$(ROOT) -I$(HERE) -I$(FAISS_INCLUDE) \
            -I$(GLOG_INCLUDE) -I$(PROTOBUF_INCLUDE)

CUDA_PATH := /usr/local/cuda-10.0
LINKFLAGS := -L$(ROOT) -L$(FAISS_LIB)  \
             -lfaiss_static -lopenblas -lgomp -lm -lquadmath -lgfortran

ifeq "$(GPU)" "OPEN"
	LINKFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas
	CXXFLAGS += -I$(CUDA_PATH)/include
endif

GXXVERSIONGTEQ5 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 5)
ifeq "$(GXXVERSIONGTEQ5)" "1"
	CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
	FAISS_RESOURCE := https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/faiss.tar.gz
endif

all:knn

####################################### knn begin ########################################
KNN_DIR := $(ROOT)/graphlearn/contrib/knn
KNN_BUILT_DIR := $(ROOT)/built/graphlearn/contrib/knn
KNN_H := $(wildcard *.h)
KNN_CC := $(wildcard *.cc)
KNN_OBJ := $(addprefix $(KNN_BUILT_DIR)/,$(patsubst %.cc,%.o,$(KNN_CC)))

$(KNN_BUILT_DIR)/%.o:$(KNN_DIR)/%.cc $(KNN_H) faiss
	@mkdir -p $(KNN_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

knn:$(KNN_OBJ)
	@mkdir -p $(ROOT)/built/lib
	$(CXX) $(CXXFLAGS) -shared $(KNN_OBJ) $(LINKFLAGS) -o $(LIB_DIR)/libknn_shared.so
####################################### knn done ########################################

faiss:
	@mkdir -p ${FAISS_DIR}
	@echo "prepare faiss library ..."
	@if [ ! -d "${FAISS_DIR}/include" ]; then cd "${FAISS_DIR}"; wget "${FAISS_RESOURCE}"; tar -zxf "$(FAISS_PKG)"; fi
	@echo "faiss done"

TEST_FLAG := -I$(GTEST_INCLUDE) -L$(GTEST_LIB) -L$(LIB_DIR) -L/lib64 -lgraphlearn_shared -lknn_shared -lgtest -lgtest_main -lstdc++ -lssl

ut:knn
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(HERE)/test/heap_unittest.cpp -o $(BIN_DIR)/heap_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/index_factory_manager_unittest.cpp -o $(BIN_DIR)/index_factory_manager_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/flat_index_unittest.cpp -o $(BIN_DIR)/flat_index_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/ivfflat_index_unittest.cpp -o $(BIN_DIR)/ivfflat_index_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/ivfpq_index_unittest.cpp -o $(BIN_DIR)/ivfpq_index_unittest $(TEST_FLAG)
