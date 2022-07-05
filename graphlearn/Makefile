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
ROOT := $(realpath $(HERE))
BUILT_DIR := $(ROOT)/built
BUILT_LIB_DIR := $(BUILT_DIR)/lib
BUILT_BIN_DIR := $(BUILT_DIR)/bin
THIRD_PARTY_DIR := $(ROOT)/../third_party
SETUP_DIR := $(ROOT)/setup
PYTHON_DIR := $(ROOT)/python

default: so

clean:
	@rm -rf $(BUILT_DIR)
	@rm -rf $(ROOT)/build/
	@rm -rf $(ROOT)/dist/
	@rm -rf $(ROOT)/graphlearn.egg-info/

cleanall:
	@rm -rf $(BUILT_DIR)
	@rm -rf $(ROOT)/build/
	@rm -rf $(ROOT)/dist/
	@rm -rf $(ROOT)/graphlearn.egg-info/
	@rm -rf ${GRPC_DIR}/build/
	@rm -rf ${GLOG_DIR}/build/
	@rm -rf ${GTEST_DIR}/build/

# grpc
GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include
GRPC_LIB := $(GRPC_DIR)/build/lib
PROTOC := $(GRPC_DIR)/build/bin/protoc
PROTOC_GRPC_PLUGIN := $(GRPC_DIR)/build/bin/grpc_cpp_plugin
grpc:
	@echo "-- installing grpc ..."
	@if [ ! -f "$(GRPC_DIR)/build/include/grpc++/grpc++.h" ]; then pushd "$(GRPC_DIR)"; /bin/bash build.sh; popd; fi
	@echo "-- grpc installed"

# glog
GLOG_DIR := $(THIRD_PARTY_DIR)/glog
GLOG_INCLUDE := $(GLOG_DIR)/build/include
GLOG_LIB := $(GLOG_DIR)/build/lib
glog:
	@echo "-- installing glog ..."
	@if [ ! -f "$(GLOG_DIR)/build/include/glog/logging.h" ]; then pushd "$(GLOG_DIR)"; /bin/bash build.sh; popd; fi
	@echo "-- glog installed"

# gtest
GTEST_DIR := $(THIRD_PARTY_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/build/include
GTEST_LIB := $(GTEST_DIR)/build/lib
gtest:
	@echo "-- installing googletest ..."
	@if [ ! -f "$(GTEST_DIR)/build/include/gtest/gtest.h" ]; then pushd "$(GTEST_DIR)"; /bin/bash build.sh; popd; fi
	@echo "-- googletest installed"

# compiling flags
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	MODEFLAGS := -DDEBUG -g
else
	MODEFLAGS := -DNDEBUG -O2
endif

KNN := OPEN
GPU := CLOSE
PROFILING := CLOSE

CXX := g++
CXXSTD := c++11
CXXFLAGS := $(MODEFLAGS) -std=$(CXXSTD) -fPIC \
            -fvisibility-inlines-hidden       \
            -pthread -mavx -msse4.2 -msse4.1  \
            -Wno-format-security              \
            -D$(KNN)_KNN                      \
            -D$(GPU)_GPU                      \
            -D$(PROFILING)_PROFILING          \
            -I$(THIRD_PARTY_DIR)              \
            -I$(ROOT)/src                     \
            -I$(GLOG_INCLUDE)                 \
            -I$(GRPC_INCLUDE)                 \

LINKFLAGS := -L$(BUILT_LIB_DIR)   \
             -L$(GLOG_LIB) -lglog \
             -L$(GRPC_LIB) -lprotobuf -lgrpc++ -lgrpc \
             -lssl -lz

FAISS_DIR := $(ROOT)/src/contrib/knn/faiss
FAISS_INCLUDE := $(FAISS_DIR)/include
FAISS_LIB := $(FAISS_DIR)/lib
ifeq "$(KNN)" "OPEN"
	CXXFLAGS += -I$(FAISS_INCLUDE)
	LINKFLAGS += -L$(FAISS_LIB) -lfaiss_static -lopenblas -lgomp -lm -lquadmath -lgfortran
endif

CUDA_PATH := /usr/local/cuda-10.0
ifeq "$(GPU)" "OPEN"
	CXXFLAGS += -I$(CUDA_PATH)/include
	LINKFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas
endif

# c++ shared library
so: grpc glog gtest proto common platform service core contrib
	@mkdir -p $(BUILT_LIB_DIR)
	@mkdir -p $(BUILT_BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared  \
		$(PROTO_OBJ) $(COMMON_OBJ) $(PLATFORM_OBJ) $(SERVICE_OBJ) $(CORE_OBJ) $(KNN_OBJ) \
		$(LINKFLAGS) -o $(BUILT_LIB_DIR)/libgraphlearn_shared.so


####################################### proto begin ########################################
PROTO_SRC_DIR := $(ROOT)/proto
PROTO_GEN_DIR := $(ROOT)/src/generated/proto
PROTO_BUILT_DIR := $(BUILT_DIR)/proto
PROTO_FILES := $(wildcard proto/*.proto)
PROTO_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.proto,%.pb.o,$(PROTO_FILES))) $(PROTO_BUILT_DIR)/service.grpc.pb.o

proto: $(PROTO_OBJ)
	@echo $(PROTO_FILES)
	@mkdir -p $(PROTO_GEN_DIR)
	@mkdir -p $(PROTO_BUILT_DIR)
	@echo 'generating pb file'
	@$(PROTOC) -I $(PROTO_SRC_DIR) --cpp_out=$(PROTO_GEN_DIR) $(PROTO_SRC_DIR)/tensor.proto
	@$(PROTOC) -I $(PROTO_SRC_DIR) --cpp_out=$(PROTO_GEN_DIR) $(PROTO_SRC_DIR)/dag.proto
	@$(PROTOC) -I $(PROTO_SRC_DIR) --cpp_out=$(PROTO_GEN_DIR) $(PROTO_SRC_DIR)/request.proto
	@$(PROTOC) -I $(PROTO_SRC_DIR) --cpp_out=$(PROTO_GEN_DIR) $(PROTO_SRC_DIR)/service.proto
	@$(PROTOC) -I $(PROTO_SRC_DIR) --grpc_out=$(PROTO_GEN_DIR) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) $(PROTO_SRC_DIR)/service.proto
	$(CXX) $(CXXFLAGS) -c $(PROTO_GEN_DIR)/tensor.pb.cc -o $(PROTO_BUILT_DIR)/tensor.pb.o
	$(CXX) $(CXXFLAGS) -c $(PROTO_GEN_DIR)/dag.pb.cc -o $(PROTO_BUILT_DIR)/dag.pb.o
	$(CXX) $(CXXFLAGS) -c $(PROTO_GEN_DIR)/request.pb.cc -o $(PROTO_BUILT_DIR)/request.pb.o
	$(CXX) $(CXXFLAGS) -c $(PROTO_GEN_DIR)/service.pb.cc -o $(PROTO_BUILT_DIR)/service.pb.o
	$(CXX) $(CXXFLAGS) -c $(PROTO_GEN_DIR)/service.grpc.pb.cc -o $(PROTO_BUILT_DIR)/service.grpc.pb.o
	@echo 'generating pb file done'
####################################### proto done ########################################

####################################### common begin ########################################
COMMON_DIR := $(ROOT)/src/common
COMMON_BUILT_DIR := $(BUILT_DIR)/src/common
COMMON_DIRS := $(shell find "src/common" -maxdepth 3 -type d)
COMMON_H := $(foreach dir,$(COMMON_DIRS),$(wildcard $(dir)/*.h))
COMMON_CC := $(foreach dir,$(COMMON_DIRS),$(wildcard $(dir)/*.cc))
COMMON_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.cc,%.o,$(COMMON_CC)))

$(COMMON_BUILT_DIR)/%.o:$(COMMON_DIR)/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/base/%.o:$(COMMON_DIR)/base/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/base
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/io/%.o:$(COMMON_DIR)/io/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/rpc/%.o:$(COMMON_DIR)/rpc/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/rpc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/string/%.o:$(COMMON_DIR)/string/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/string
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/%.o:$(COMMON_DIR)/threading/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/atomic/%.o:$(COMMON_DIR)/threading/atomic/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading/atomic
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/lockfree/%.o:$(COMMON_DIR)/threading/lockfree/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading/lockfree
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/runner/%.o:$(COMMON_DIR)/threading/runner/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading/runner
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/sync/%.o:$(COMMON_DIR)/threading/sync/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading/sync
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_BUILT_DIR)/threading/thread/%.o:$(COMMON_DIR)/threading/thread/%.cc $(COMMON_H)
	@mkdir -p $(COMMON_BUILT_DIR)/threading/thread
	$(CXX) $(CXXFLAGS) -c $< -o $@

common:$(COMMON_OBJ)
####################################### common done ########################################

####################################### platform begin ########################################
PLATFORM_DIR := $(ROOT)/src/platform
PLATFORM_BUILT_DIR := $(BUILT_DIR)/src/platform
PLATFORM_DIRS := src/platform src/platform/local src/platform/hadoop
PLATFORM_H := $(foreach dir,$(PLATFORM_DIRS),$(wildcard $(dir)/*.h))
PLATFORM_CC := $(foreach dir,$(PLATFORM_DIRS),$(wildcard $(dir)/*.cc))
PLATFORM_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.cc,%.o,$(PLATFORM_CC)))

$(PLATFORM_BUILT_DIR)/%.o:$(PLATFORM_DIR)/%.cc $(PLATFORM_H)
	@mkdir -p $(PLATFORM_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PLATFORM_BUILT_DIR)/local/%.o:$(PLATFORM_DIR)/local/%.cc $(PLATFORM_H)
	@mkdir -p $(PLATFORM_BUILT_DIR)/local
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PLATFORM_BUILT_DIR)/hadoop/%.o:$(PLATFORM_DIR)/hadoop/%.cc $(PLATFORM_H)
	@mkdir -p $(PLATFORM_BUILT_DIR)/hadoop
	$(CXX) $(CXXFLAGS) -c $< -o $@

platform:$(PLATFORM_OBJ)
####################################### platform done ########################################

####################################### service begin ########################################
SERVICE_DIR := $(ROOT)/src/service
SERVICE_BUILT_DIR := $(BUILT_DIR)/src/service
SERVICE_DIRS := $(shell find "src/service" -maxdepth 3 -type d)
SERVICE_H := $(foreach dir,$(SERVICE_DIRS),$(wildcard $(dir)/*.h))
SERVICE_CC := $(foreach dir,$(SERVICE_DIRS),$(wildcard $(dir)/*.cc))
SERVICE_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.cc,%.o,$(SERVICE_CC)))

$(SERVICE_BUILT_DIR)/%.o:$(SERVICE_DIR)/%.cc $(SERVICE_H)
	@mkdir -p $(SERVICE_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SERVICE_BUILT_DIR)/local/%.o:$(SERVICE_DIR)/local/%.cc $(SERVICE_H)
	@mkdir -p $(SERVICE_BUILT_DIR)/local
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SERVICE_BUILT_DIR)/dist/%.o:$(SERVICE_DIR)/dist/%.cc $(SERVICE_H)
	@mkdir -p $(SERVICE_BUILT_DIR)/dist
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SERVICE_BUILT_DIR)/request/%.o:$(SERVICE_DIR)/request/%.cc $(SERVICE_H)
	@mkdir -p $(SERVICE_BUILT_DIR)/request
	$(CXX) $(CXXFLAGS) -c $< -o $@

service:$(SERVICE_OBJ)
####################################### service done ########################################

####################################### core begin ########################################
CORE_DIR := $(ROOT)/src/core
CORE_BUILT_DIR := $(BUILT_DIR)/src/core
CORE_DIRS := $(shell find "src/core" -maxdepth 3 -type d)
CORE_H := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.h))
CORE_CC := $(foreach dir,$(CORE_DIRS),$(wildcard $(dir)/*.cc))
CORE_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.cc,%.o,$(CORE_CC)))

$(CORE_BUILT_DIR)/%.o:$(CORE_DIR)/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/graph/%.o:$(CORE_DIR)/graph/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/graph
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/graph/storage/%.o:$(CORE_DIR)/graph/storage/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/graph/storage
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/io/%.o:$(CORE_DIR)/io/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/%.o:$(CORE_DIR)/operator/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/graph/%.o:$(CORE_DIR)/operator/graph/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/graph
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/sampler/%.o:$(CORE_DIR)/operator/sampler/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/sampler
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/sampler/padder/%.o:$(CORE_DIR)/operator/sampler/padder/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/sampler/padder
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/aggregator/%.o:$(CORE_DIR)/operator/aggregator/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/aggregator
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/subgraph/%.o:$(CORE_DIR)/operator/subgraph/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/subgraph
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/operator/utils/%.o:$(CORE_DIR)/operator/utils/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/operator/utils
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/partition/%.o:$(CORE_DIR)/partition/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/partition
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/runner/%.o:$(CORE_DIR)/runner/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/runner
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CORE_BUILT_DIR)/dag/%.o:$(CORE_DIR)/dag/%.cc $(CORE_H)
	@mkdir -p $(CORE_BUILT_DIR)/dag
	$(CXX) $(CXXFLAGS) -c $< -o $@

core:$(CORE_OBJ)
####################################### core done ########################################

####################################### contrib begin ########################################
CONTRIB_DIR := $(ROOT)/src/contrib

faiss:
	@mkdir -p ${FAISS_DIR}
	@echo "download faiss library ..."
	@if [ ! -d "${FAISS_DIR}/include" ]; then cd "${FAISS_DIR}"; wget "${FAISS_RESOURCE}"; tar -zxf "$(FAISS_PKG)"; fi
	@echo "faiss done"

KNN_DIR := $(CONTRIB_DIR)/knn
KNN_BUILT_DIR := $(BUILT_DIR)/src/contrib/knn

ifeq "$(KNN)" "OPEN"
KNN_H := $(wildcard *.h)
KNN_CC := $(wildcard *.cc)
KNN_OBJ := $(addprefix $(KNN_BUILT_DIR)/,$(patsubst %.cc,%.o,$(KNN_CC)))

$(KNN_BUILT_DIR)/%.o:$(KNN_DIR)/%.cc $(KNN_H) faiss
	@mkdir -p $(KNN_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else
KNN_OBJ :=
endif

knn:$(KNN_OBJ)

contrib:knn
####################################### contrib done ########################################

TEST_FLAG := -I$(GTEST_INCLUDE) -L$(GTEST_LIB) -L$(BUILT_LIB_DIR) -L$(GRPC_LIB) -L/lib64 \
             -lgtest -lgtest_main -lstdc++ -lssl -lz
TEST_FLAG += -lgraphlearn_shared

test:so gtest
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/base/test/closure_unittest.cpp -o $(BUILT_BIN_DIR)/closure_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/base/test/status_unittest.cpp -o $(BUILT_BIN_DIR)/status_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/atomic/test/atomic_unittest.cpp -o $(BUILT_BIN_DIR)/atomic_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/lockfree/test/lockfree_queue_unittest.cpp -o $(BUILT_BIN_DIR)/lockfree_queue_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/lockfree/test/lockfree_stack_unittest.cpp -o $(BUILT_BIN_DIR)/lockfree_stack_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/runner/test/dynamic_worker_thread_pool_unittest.cpp -o $(BUILT_BIN_DIR)/dynamic_worker_thread_pool_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/sync/test/cond_unittest.cpp -o $(BUILT_BIN_DIR)/cond_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/sync/test/lock_unittest.cpp -o $(BUILT_BIN_DIR)/lock_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/sync/test/waitable_event_unittest.cpp -o $(BUILT_BIN_DIR)/waitable_event_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/test/this_thread_unittest.cpp -o $(BUILT_BIN_DIR)/this_thread_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(COMMON_DIR)/threading/thread/test/thread_unittest.cpp -o $(BUILT_BIN_DIR)/thread_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/graph/test/graph_store_unittest.cpp -o $(BUILT_BIN_DIR)/graph_store_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/graph/storage/test/node_storage_unittest.cpp -o $(BUILT_BIN_DIR)/node_storage_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/graph/storage/test/graph_storage_unittest.cpp -o $(BUILT_BIN_DIR)/graph_storage_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/io/test/data_slicer_unittest.cpp -o $(BUILT_BIN_DIR)/data_slicer_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/io/test/edge_loader_unittest.cpp -o $(BUILT_BIN_DIR)/edge_loader_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/io/test/node_loader_unittest.cpp -o $(BUILT_BIN_DIR)/node_loader_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/operator/graph/test/graph_op_unittest.cpp -o $(BUILT_BIN_DIR)/graph_op_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/operator/sampler/test/sampler_unittest.cpp -o $(BUILT_BIN_DIR)/sampler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/operator/sampler/test/negative_sampler_unittest.cpp -o $(BUILT_BIN_DIR)/negative_sampler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/operator/aggregator/test/aggregating_op_unittest.cpp -o $(BUILT_BIN_DIR)/aggregating_op_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(CORE_DIR)/runner/test/thread_dag_scheduler_unittest.cpp -o $(BUILT_BIN_DIR)/thread_dag_scheduler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(PLATFORM_DIR)/test/env_unittest.cpp -o $(BUILT_BIN_DIR)/env_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(PLATFORM_DIR)/test/local_fs_unittest.cpp -o $(BUILT_BIN_DIR)/local_fs_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/test/event_queue_unittest.cpp -o $(BUILT_BIN_DIR)/event_queue_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/test/tensor_unittest.cpp -o $(BUILT_BIN_DIR)/tensor_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/test/client_test.cpp -o $(BUILT_BIN_DIR)/client_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/test/server_test.cpp -o $(BUILT_BIN_DIR)/server_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/test/dist_in_memory_test.cpp -o $(BUILT_BIN_DIR)/dist_in_memory_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/request/test/graph_request_unittest.cpp -o $(BUILT_BIN_DIR)/graph_request_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/request/test/aggregating_request_unittest.cpp -o $(BUILT_BIN_DIR)/aggregating_request_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/dist/test/naming_engine_unittest.cpp -o $(BUILT_BIN_DIR)/naming_engine_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/dist/test/coordinator_unittest.cpp -o $(BUILT_BIN_DIR)/coordinator_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/dist/test/channel_manager_unittest.cpp -o $(BUILT_BIN_DIR)/channel_manager_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/dist/test/service_unittest.cpp -o $(BUILT_BIN_DIR)/service_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(SERVICE_DIR)/dist/test/service_with_hosts_unittest.cpp -o $(BUILT_BIN_DIR)/service_with_hosts_unittest $(TEST_FLAG)
	ifeq "$(KNN)" "OPEN"
		$(CXX) $(CXXFLAGS) $(KNN_DIR)/test/heap_unittest.cpp -o $(BUILT_BIN_DIR)/heap_unittest $(TEST_FLAG)
		$(CXX) $(CXXFLAGS) $(KNN_DIR)/test/index_factory_manager_unittest.cpp -o $(BUILT_BIN_DIR)/index_factory_manager_unittest $(TEST_FLAG)
		$(CXX) $(CXXFLAGS) $(KNN_DIR)/test/flat_index_unittest.cpp -o $(BUILT_BIN_DIR)/flat_index_unittest $(TEST_FLAG)
		$(CXX) $(CXXFLAGS) $(KNN_DIR)/test/ivfflat_index_unittest.cpp -o $(BUILT_BIN_DIR)/ivfflat_index_unittest $(TEST_FLAG)
		$(CXX) $(CXXFLAGS) $(KNN_DIR)/test/ivfpq_index_unittest.cpp -o $(BUILT_BIN_DIR)/ivfpq_index_unittest $(TEST_FLAG)
	endif

VERSION := $(shell grep "_VERSION = " ${SETUP_DIR}/setup.py | cut -d= -f2)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_VERSION := $(shell git rev-parse --short HEAD)
PYTHON := python
python: so
	@rm -rf dist
	@rm -rf graphlearn.egg-info
	@cp $(SETUP_DIR)/gl.__init__.py $(PYTHON_DIR)/__init__.py
	@echo "__version__ = $(VERSION)" >> $(PYTHON_DIR)/__init__.py
	@echo "__git_version__ = '$(GIT_BRANCH)-$(GIT_VERSION)'" >> $(PYTHON_DIR)/__init__.py

#	@if [ "$(KNN)" = "OPEN" ]; then cp $(BUILT_LIB_DIR)/libknn_shared.so $(PYTHON_LIB); fi

	OPEN_KNN=${KNN} ${PYTHON} $(SETUP_DIR)/setup.py bdist_wheel
	@mkdir -p $(BUILT_BIN_DIR)/ge_data/data
	@mkdir -p $(BUILT_BIN_DIR)/ge_data/ckpt
#	@rm -rf $(PYTHON_DIR)/__init__.py*
