HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := $(realpath $(HERE))
BUILT_DIR := $(ROOT)/built
INCLUDE_DIR := $(BUILT_DIR)/include
LIB_DIR := $(BUILT_DIR)/lib
BIN_DIR := $(BUILT_DIR)/bin
THIRD_PARTY_DIR := $(ROOT)/third_party
SETUP_DIR := $(ROOT)/setup
PYTHON_DIR := $(ROOT)/graphlearn
PYTHON_LIB := $(PYTHON_DIR)/python/lib

default: so

clean:
	@rm -rf $(BUILT_DIR)
	@rm -rf $(PYTHON_LIB)
	@rm -rf build/
	@rm -rf dist/
	@rm -rf graphlearn.egg-info/

cleanall:
	@rm -rf $(BUILT_DIR)
	@rm -rf build/
	@rm -rf dist/
	@rm -rf graphlearn.egg-info/
	@rm -rf ${PROTOBUF_DIR}/build/
	@rm -rf ${PROTOBUF_DIR}/protobuf/
	@rm -rf ${GRPC_DIR}/build/
	@rm -rf ${GRPC_DIR}/grpc/
	@rm -rf ${GLOG_DIR}/build/
	@rm -rf ${GLOG_DIR}/glog/
	@rm -rf ${PYBIND_DIR}/build/
	@rm -rf ${PYBIND_DIR}/pybind11/
	@rm -rf ${GTEST_DIR}/build/
	@rm -rf ${GTEST_DIR}/googletest/

# protobuf
PROTOBUF_DIR := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_INCLUDE := $(PROTOBUF_DIR)/build/include
PROTOBUF_LIB := $(PROTOBUF_DIR)/build/lib
PROTOC := $(PROTOBUF_DIR)/build/bin/protoc
protobuf:
	@echo "prepare protobuf library ..."
	@if [ ! -d "${PROTOBUF_DIR}/build" ]; then cd "${PROTOBUF_DIR}"; ./build.sh; fi
	@echo "protobuf done"

# grpc
GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include
GRPC_LIB := $(GRPC_DIR)/build/lib
PROTOC_GRPC_PLUGIN := $(GRPC_DIR)/build/bin/grpc_cpp_plugin
grpc:
	@echo "prepare grpc library ..."
	@if [ ! -d "${GRPC_DIR}/build" ]; then cd "${GRPC_DIR}"; ./build.sh; fi
	@echo "grpc done"

# glog
GLOG_DIR := $(THIRD_PARTY_DIR)/glog
GLOG_INCLUDE := $(GLOG_DIR)/build
GLOG_LIB := $(GLOG_DIR)/build
glog:
	@echo "prepare glog library ..."
	@if [ ! -d "${GLOG_DIR}/build" ]; then cd "${GLOG_DIR}"; ./build.sh; fi
	@echo "glog done"

# pybind11
PYBIND_DIR := $(THIRD_PARTY_DIR)/pybind11
PYBIND_INCLUDE := $(PYBIND_DIR)/pybind11/include/
pybind:
	@echo "prepare pybind11 library ..."
	@if [ ! -d "${PYBIND_DIR}/build" ]; then cd "${PYBIND_DIR}"; ./build.sh; fi
	@echo "pybind11 done"

# gtest
GTEST_DIR := $(THIRD_PARTY_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/googletest/googletest/include
GTEST_LIB := $(GTEST_DIR)/build/googlemock/gtest
gtest:
	@echo "prepare gtest library ..."
	@if [ ! -d "${GTEST_DIR}/build" ]; then cd "${GTEST_DIR}"; ./build.sh; fi
	@echo "gtest done"

# compling flags
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	MODEFLAGS := -DDEBUG -g
else
	MODEFLAGS := -DNDEBUG -O2
endif

PROFILING := CLOSE

CXX := g++
CXXSTD := c++11
CXXFLAGS := $(MODEFLAGS) -std=$(CXXSTD) -fPIC \
            -fvisibility-inlines-hidden       \
            -pthread -mavx -msse4.2 -msse4.1  \
            -D$(PROFILING)_PROFILING          \
            -I. -I$(ROOT) -I$(BUILT_DIR)      \
            -I$(PROTOBUF_INCLUDE)             \
            -I$(GLOG_INCLUDE)                 \
            -I$(GRPC_INCLUDE)                 \

LINKFLAGS := -L$(ROOT) -L$(LIB_DIR)                        \
             -L$(GLOG_LIB) -L$(PROTOBUF_LIB) -L$(GRPC_LIB) \
             -lglog -lprotobuf -lgrpc++ -lgrpc -lssl -lz

# c++ shared library
so: protobuf grpc glog gtest proto common platform service core
	@mkdir -p $(INCLUDE_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared  \
		$(PROTO_OBJ) $(COMMON_OBJ) $(PLATFORM_OBJ) $(SERVICE_OBJ) $(CORE_OBJ) \
		$(LINKFLAGS) -o $(LIB_DIR)/libgraphlearn_shared.so


####################################### proto begin ########################################
PROTO_DIR := $(ROOT)/graphlearn/proto
PROTO_BUILT_DIR := $(BUILT_DIR)/graphlearn/proto
PROTO_DIRS := $(shell find "graphlearn/proto" -maxdepth 3 -type d)
PROTO_FILES := $(foreach dir,$(PROTO_DIRS),$(wildcard $(dir)/*.proto))
PROTO_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.proto,%.pb.o,$(PROTO_FILES))) $(PROTO_BUILT_DIR)/service.grpc.pb.o

proto: $(PROTO_FILES)
	@echo $(PROTO_FILES)
	@echo $(PROTO_OBJ)
	@mkdir -p $(PROTO_BUILT_DIR)
	@echo 'generating pb file'
	@$(PROTOC) --cpp_out=. graphlearn/proto/tensor.proto
	@$(PROTOC) --cpp_out=. graphlearn/proto/dag.proto
	@$(PROTOC) --cpp_out=. graphlearn/proto/request.proto
	@$(PROTOC) --cpp_out=. graphlearn/proto/service.proto
	@$(PROTOC) --grpc_out=. --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) graphlearn/proto/service.proto
	@$(CXX) $(CXXFLAGS) -c $(PROTO_DIR)/tensor.pb.cc -o $(PROTO_BUILT_DIR)/tensor.pb.o
	@$(CXX) $(CXXFLAGS) -c $(PROTO_DIR)/dag.pb.cc -o $(PROTO_BUILT_DIR)/dag.pb.o
	@$(CXX) $(CXXFLAGS) -c $(PROTO_DIR)/request.pb.cc -o $(PROTO_BUILT_DIR)/request.pb.o
	@$(CXX) $(CXXFLAGS) -c $(PROTO_DIR)/service.pb.cc -o $(PROTO_BUILT_DIR)/service.pb.o
	@$(CXX) $(CXXFLAGS) -c $(PROTO_DIR)/service.grpc.pb.cc -o $(PROTO_BUILT_DIR)/service.grpc.pb.o
	@echo 'generating pb file done'
####################################### proto done ########################################

####################################### common begin ########################################
COMMON_DIR := $(ROOT)/graphlearn/common
COMMON_BUILT_DIR := $(BUILT_DIR)/graphlearn/common
COMMON_DIRS := $(shell find "graphlearn/common" -maxdepth 3 -type d)
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
PLATFORM_DIR := $(ROOT)/graphlearn/platform
PLATFORM_BUILT_DIR := $(BUILT_DIR)/graphlearn/platform
PLATFORM_DIRS := graphlearn/platform graphlearn/platform/local
PLATFORM_H := $(foreach dir,$(PLATFORM_DIRS),$(wildcard $(dir)/*.h))
PLATFORM_CC := $(foreach dir,$(PLATFORM_DIRS),$(wildcard $(dir)/*.cc))
PLATFORM_OBJ := $(addprefix $(BUILT_DIR)/,$(patsubst %.cc,%.o,$(PLATFORM_CC)))

$(PLATFORM_BUILT_DIR)/%.o:$(PLATFORM_DIR)/%.cc $(PLATFORM_H)
	@mkdir -p $(PLATFORM_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PLATFORM_BUILT_DIR)/local/%.o:$(PLATFORM_DIR)/local/%.cc $(PLATFORM_H)
	@mkdir -p $(PLATFORM_BUILT_DIR)/local
	$(CXX) $(CXXFLAGS) -c $< -o $@

platform:$(PLATFORM_OBJ)
####################################### platform done ########################################

####################################### service begin ########################################
SERVICE_DIR := $(ROOT)/graphlearn/service
SERVICE_BUILT_DIR := $(BUILT_DIR)/graphlearn/service
SERVICE_DIRS := $(shell find "graphlearn/service" -maxdepth 3 -type d)
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
CORE_DIR := $(ROOT)/graphlearn/core
CORE_BUILT_DIR := $(BUILT_DIR)/graphlearn/core
CORE_DIRS := $(shell find "graphlearn/core" -maxdepth 3 -type d)
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

TEST_FLAG := -I$(GTEST_INCLUDE) -L$(GTEST_LIB) -L$(LIB_DIR) -L$(GRPC_LIB) -L/lib64 \
             -lgraphlearn_shared -lgtest -lgtest_main -lstdc++ -lssl -lz

test:so gtest
	$(CXX) $(CXXFLAGS) graphlearn/common/base/test/closure_unittest.cpp -o built/bin/closure_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/base/test/status_unittest.cpp -o built/bin/status_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/atomic/test/atomic_unittest.cpp -o built/bin/atomic_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/lockfree/test/lockfree_queue_unittest.cpp -o built/bin/lockfree_queue_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/lockfree/test/lockfree_stack_unittest.cpp -o built/bin/lockfree_stack_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/runner/test/dynamic_worker_thread_pool_unittest.cpp -o built/bin/dynamic_worker_thread_pool_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/sync/test/cond_unittest.cpp -o built/bin/cond_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/sync/test/lock_unittest.cpp -o built/bin/lock_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/sync/test/waitable_event_unittest.cpp -o built/bin/waitable_event_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/test/this_thread_unittest.cpp -o built/bin/this_thread_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/common/threading/thread/test/thread_unittest.cpp -o built/bin/thread_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/graph/test/graph_store_unittest.cpp -o built/bin/graph_store_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/graph/storage/test/node_storage_unittest.cpp -o built/bin/node_storage_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/graph/storage/test/graph_storage_unittest.cpp -o built/bin/graph_storage_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/io/test/data_slicer_unittest.cpp -o built/bin/data_slicer_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/io/test/edge_loader_unittest.cpp -o built/bin/edge_loader_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/io/test/node_loader_unittest.cpp -o built/bin/node_loader_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/operator/graph/test/graph_op_unittest.cpp -o built/bin/graph_op_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/operator/sampler/test/sampler_unittest.cpp -o built/bin/sampler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/operator/sampler/test/negative_sampler_unittest.cpp -o built/bin/negative_sampler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/operator/aggregator/test/aggregating_op_unittest.cpp -o built/bin/aggregating_op_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/core/runner/test/thread_dag_scheduler_unittest.cpp -o built/bin/thread_dag_scheduler_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/platform/test/env_unittest.cpp -o built/bin/env_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/platform/test/local_fs_unittest.cpp -o built/bin/local_fs_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/test/event_queue_unittest.cpp -o built/bin/event_queue_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/test/tensor_unittest.cpp -o built/bin/tensor_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/test/client_test.cpp -o built/bin/client_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/test/server_test.cpp -o built/bin/server_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/test/dist_in_memory_test.cpp -o built/bin/dist_in_memory_test $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/request/test/graph_request_unittest.cpp -o built/bin/graph_request_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/request/test/aggregating_request_unittest.cpp -o built/bin/aggregating_request_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/dist/test/naming_engine_unittest.cpp -o built/bin/naming_engine_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/dist/test/coordinator_unittest.cpp -o built/bin/coordinator_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/dist/test/channel_manager_unittest.cpp -o built/bin/channel_manager_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/dist/test/service_unittest.cpp -o built/bin/service_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) graphlearn/service/dist/test/service_with_hosts_unittest.cpp -o built/bin/service_with_hosts_unittest $(TEST_FLAG)

VERSION := $(shell grep "_VERSION = " ${SETUP_DIR}/setup.py | cut -d= -f2)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_VERSION := $(shell git rev-parse --short HEAD)
PYTHON := python
python: so pybind
	@rm -rf build
	@rm -rf dist
	@rm -rf graphlearn.egg-info
	@rm -rf $(PYTHON_LIB)
	@mkdir -p $(PYTHON_LIB)
	@cp $(SETUP_DIR)/gl.__init__.py $(PYTHON_DIR)/__init__.py
	@echo "__version__ = \"$(VERSION)\"" >> $(PYTHON_DIR)/__init__.py
	@echo "__git_version__ = '$(GIT_BRANCH)-$(GIT_VERSION)'" >> $(PYTHON_DIR)/__init__.py
	@cp $(LIB_DIR)/libgraphlearn_shared.so $(PYTHON_LIB)

	${PYTHON} $(SETUP_DIR)/setup.py bdist_wheel
	@mkdir -p $(BIN_DIR)/ge_data/data
	@mkdir -p $(BIN_DIR)/ge_data/ckpt
	@rm -rf $(PYTHON_DIR)/__init__.py*
