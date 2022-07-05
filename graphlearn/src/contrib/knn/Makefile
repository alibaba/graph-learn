# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := ../../..
BUILT_DIR := $(ROOT)/built
BUILT_LIB_DIR := $(BUILT_DIR)/lib
BUILT_BIN_DIR := $(BUILT_DIR)/bin
THIRD_PARTY_DIR := $(ROOT)/../third_party

GLOG_DIR := $(THIRD_PARTY_DIR)/glog
GLOG_INCLUDE := $(GLOG_DIR)/build

GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include

GTEST_DIR := $(THIRD_PARTY_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/build/include
GTEST_LIB := $(GTEST_DIR)/build/lib

FAISS_PKG := faiss_gcc4.8.5.tar.gz
FAISS_RESOURCE := https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/$(FAISS_PKG)
FAISS_DIR := $(HERE)/faiss
FAISS_INCLUDE := $(FAISS_DIR)/include
FAISS_LIB := $(FAISS_DIR)/lib

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	MODEFLAGS := -DDEBUG -g
else
	MODEFLAGS := -DNDEBUG -O2
endif

GPU := CLOSE
CXX := g++
CXXSTD := c++11
CXXFLAGS := $(MODEFLAGS) -std=$(CXXSTD) -fPIC    \
            -pthread -mavx -msse4.2 -msse4.1     \
            -Wno-format-security                 \
            -D$(GPU)_GPU                         \
            -I$(ROOT)/src -I$(FAISS_INCLUDE)     \
            -I$(GLOG_INCLUDE) -I$(GRPC_INCLUDE)

LINKFLAGS := -L$(ROOT) -L$(ROOT)/built/lib -L$(FAISS_LIB)  \
             -lfaiss_static -lopenblas -lgomp -lm -lquadmath -lgfortran

CUDA_PATH := /usr/local/cuda-10.0
ifeq "$(GPU)" "OPEN"
	LINKFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas
	CXXFLAGS += -I$(CUDA_PATH)/include
endif

GXXVERSIONGTEQ5 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 5)
ifeq "$(GXXVERSIONGTEQ5)" "1"
	FAISS_PKG := faiss.tar.gz
	FAISS_RESOURCE := https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/faiss.tar.gz
endif

all:knn

####################################### knn begin ########################################
KNN_DIR := $(ROOT)/src/contrib/knn
KNN_BUILT_DIR := $(ROOT)/built/src/contrib/knn
KNN_H := $(wildcard *.h)
KNN_CC := $(wildcard *.cc)
KNN_OBJ := $(addprefix $(KNN_BUILT_DIR)/,$(patsubst %.cc,%.o,$(KNN_CC)))

$(KNN_BUILT_DIR)/%.o:$(KNN_DIR)/%.cc $(KNN_H) faiss
	@mkdir -p $(KNN_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

knn:$(KNN_OBJ)
	@mkdir -p $(ROOT)/built/lib
	$(CXX) $(CXXFLAGS) -shared $(KNN_OBJ) $(LINKFLAGS) -o $(BUILT_LIB_DIR)/libknn_shared.so
####################################### knn done ########################################

faiss:
	@mkdir -p ${FAISS_DIR}
	@echo "download faiss library ..."
	@if [ ! -d "${FAISS_DIR}/include" ]; then cd "${FAISS_DIR}"; wget "${FAISS_RESOURCE}"; tar -zxf "$(FAISS_PKG)"; rm -f "$(FAISS_PKG)"; fi
	@echo "faiss done"

TEST_FLAG := -I$(GTEST_INCLUDE) -L$(GTEST_LIB) -L$(BUILT_LIB_DIR) -L/lib64 -lknn_shared -lgtest -lgtest_main -lstdc++ -lssl

ut:knn
	@mkdir -p $(BUILT_BIN_DIR)
	$(CXX) $(CXXFLAGS) $(HERE)/test/heap_unittest.cpp -o $(BUILT_BIN_DIR)/heap_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/index_factory_manager_unittest.cpp -o $(BUILT_BIN_DIR)/index_factory_manager_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/flat_index_unittest.cpp -o $(BUILT_BIN_DIR)/flat_index_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/ivfflat_index_unittest.cpp -o $(BUILT_BIN_DIR)/ivfflat_index_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/test/ivfpq_index_unittest.cpp -o $(BUILT_BIN_DIR)/ivfpq_index_unittest $(TEST_FLAG)
