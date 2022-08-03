HERE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT := ../..
BUILT_DIR := $(ROOT)/built
LIB_DIR := $(BUILT_DIR)/lib
BIN_DIR := $(BUILT_DIR)/bin
THIRD_PARTY_DIR := $(ROOT)/third_party
PYTHON_LIB := $(ROOT)/graphlearn/python/lib

# protobuf
PROTOBUF_DIR := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_INCLUDE := $(PROTOBUF_DIR)/build/include

# glog
GLOG_DIR := $(THIRD_PARTY_DIR)/glog
GLOG_INCLUDE := $(GLOG_DIR)/build

# grpc
GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include

# brane
BRANE_DIR := $(THIRD_PARTY_DIR)/brane
BRANE_INCLUDE := ${BRANE_DIR}/build/include
BRANE_LIB := ${BRANE_DIR}/build/lib64
BRANE_DEPS_INCLUDE := ${BRANE_DIR}/brane/build/_cooking/installed/include
BRANE_DEPS_LIB := ${BRANE_DIR}/brane/build/_cooking/installed/lib
BRANE_DEPS_LIB64 := $(THIRD_PARTY_DIR)/brane/brane/build/_cooking/installed/lib64
brane:
	@echo "prepare brane library ..."
	@if [ ! -d "${BRANE_DIR}/build" ]; then cd "${BRANE_DIR}"; ./build.sh; fi
	@echo "brane done"

# compling flags
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	MODEFLAGS := -DDEBUG -g
else
	MODEFLAGS := -DNDEBUG -O2
endif

CXX := g++
CXXFLAGS := $(MODEFLAGS) -std=gnu++14 -fPIC    \
            -pthread -mavx -msse4.2 -msse4.1   \
            -Wno-attributes                    \
            -I$(ROOT) -I$(HERE)                \
            -I$(PROTOBUF_INCLUDE)              \
            -I$(GLOG_INCLUDE)                  \
            -I$(GRPC_INCLUDE)                  \
            -I$(BRANE_INCLUDE)                 \
            -I$(BRANE_DEPS_INCLUDE)

GXXVERSIONGTEQ5 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 5)
ifeq "$(GXXVERSIONGTEQ5)" "1"
	CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif

LINKFLAGS := -L$(BRANE_LIB) -L$(BRANE_DEPS_LIB) -L$(BRANE_DEPS_LIB64) -lbrane_shared

default:actor

GLA_DIR := graphlearn/actor
ACTOR_DIR := $(ROOT)/${GLA_DIR}
ACTOR_BUILT_DIR := $(BUILT_DIR)/${GLA_DIR}
ACTOR_AUTOGEN_DIR := ${ACTOR_DIR}/generated
ACTOR_DIRS := $(filter-out generated, $(shell find * -maxdepth 3 -type d))

## get auto-generated files
ACTOR_ACT_H := $(foreach dir,$(ACTOR_DIRS),$(wildcard $(dir)/*.act.h))
AUTOGEN_FNAMES := $(notdir $(patsubst %.act.h,%.act.autogen.cc,$(ACTOR_ACT_H)))
ACTOR_AG_CC := $(addprefix generated/,$(AUTOGEN_FNAMES))

ACTOR_H := $(wildcard *.h) $(foreach dir,$(ACTOR_DIRS),$(wildcard $(dir)/*.h))
ACTOR_CC := $(wildcard *.cc) $(foreach dir,$(ACTOR_DIRS),$(wildcard $(dir)/*.cc)) $(ACTOR_AG_CC)
ACTOR_OBJ := $(addprefix $(ACTOR_BUILT_DIR)/,$(patsubst %.cc,%.o,$(ACTOR_CC)))

CODEGEN_TOOL := $(THIRD_PARTY_DIR)/brane/brane/actor-codegen-tool/actor_codegen.py
LIBCLANG_PATH := /usr/lib64

DUMMY := $(shell rm -rf $(ACTOR_DIR)/generated)
DUMMY := $(shell mkdir -p $(ACTOR_DIR)/generated)
DUMMY := $(shell for fname in $(ACTOR_AG_CC); do touch $$fname; done)

codegen:$(ACTOR_ACT_H)
	@echo "generating actor definition file"
	@python3 $(CODEGEN_TOOL) --libclang-dir=$(LIBCLANG_PATH) \
		--project-dir=$(ROOT) --actor-mod-dir=$(GLA_DIR) \
		--sysactor-include=$(BRANE_INCLUDE) \
		--user-include="$(ROOT);$(PROTOBUF_INCLUDE)" \
		--sys-include=$(BRANE_DEPS_INCLUDE)
	@echo "generating actor definition done"

$(ACTOR_BUILT_DIR)/%.o: %.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/generated/%.o: generated/%.cc $(ACTOR_H) codegen brane
	@mkdir -p $(ACTOR_BUILT_DIR)/generated
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/graph/%.o: graph/%.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)/graph
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/operator/%.o: operator/%.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)/operator
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/dag/%.o: dag/%.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)/dag
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/runner/%.o: runner/%.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)/runner
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ACTOR_BUILT_DIR)/service/%.o: service/%.cc $(ACTOR_H) brane
	@mkdir -p $(ACTOR_BUILT_DIR)/service
	$(CXX) $(CXXFLAGS) -c $< -o $@

actor:$(ACTOR_OBJ) brane codegen
	@mkdir -p $(ROOT)/built/lib
	$(CXX) $(CXXFLAGS) -shared $(ACTOR_OBJ) $(LINKFLAGS) -o $(LIB_DIR)/libactor_shared.so
	@cp $(BRANE_LIB)/libbrane_shared.so $(LIB_DIR)/

GTEST_DIR := $(THIRD_PARTY_DIR)/googletest
GTEST_INCLUDE := $(GTEST_DIR)/googletest/googletest/include
GTEST_LIB := $(GTEST_DIR)/build/googlemock/gtest

TEST_FLAG := -I$(GTEST_INCLUDE) -L$(GTEST_LIB) -L$(LIB_DIR) \
	-L$(BRANE_LIB) -L$(BRANE_DEPS_LIB) -L$(BRANE_DEPS_LIB64) -L/lib64 \
	-lbrane_shared -lgraphlearn_shared -lactor_shared \
	-lboost_system -lgtest -lgtest_main -lstdc++ -lssl

ut:
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(HERE)/graph/test/sharded_graph_store_unittest.cpp -o $(BIN_DIR)/sharded_graph_store_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/operator/test/batch_generator_unittest.cpp -o $(BIN_DIR)/batch_generator_unittest $(TEST_FLAG)
	$(CXX) $(CXXFLAGS) $(HERE)/runner/test/actor_dag_scheduler_unittest.cpp -o $(BIN_DIR)/actor_dag_scheduler_unittest $(TEST_FLAG)

lib_for_python:
	@cp $(LIB_DIR)/libactor_shared.so $(PYTHON_LIB)
	@cp $(BRANE_LIB)/libbrane_shared.so $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_system.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_program_options.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_thread.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_chrono.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_date_time.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_timer.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_atomic.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libboost_filesystem.so.1.64.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libcares.so.2 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libcryptopp.so.5.6 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/liblz4.so.1 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libgnutls.so.30 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libgnutlsxx.so.28 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libsctp.so.1 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libyaml-cpp.so.0.5 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libhogweed.so.4 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libgmp.so.10 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libcolm-0.13.0.6.so $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libhwloc.so.5 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libnettle.so.6 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libpciaccess.so.0 $(PYTHON_LIB)
	@cp $(BRANE_DEPS_LIB)/libz.so.1 $(PYTHON_LIB)

cleanall:
	@rm -rf ${BRANE_DIR}/build/
	@rm -rf ${BRANE_DIR}/brane/