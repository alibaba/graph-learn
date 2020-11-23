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

cmake_minimum_required (VERSION 3.5)

project (GraphLearn VERSION 1.0 LANGUAGES CXX)

set (CMAKE_CXX_COMPILER g++)

option (TESTING "ENABLE testing" OFF)
option (GL_DEBUG "Enable debug mode" OFF)
option (PROFILING "Enable profiling" OFF)
option (WITH_VINEYARD "Enable vineyard" OFF)
set (PYTHON_VERSION
    "python3"
    CACHE
    STRING
    "python version")

set (HERE ${CMAKE_CURRENT_SOURCE_DIR})
set (ROOT ${HERE})
set (BUILT_DIR ${ROOT}/built)
set (INCLUDE_DIR ${BUILT_DIR}/include)
set (LIB_DIR ${BUILT_DIR}/lib)
set (BIN_DIR ${BUILT_DIR}/bin)
set (THIRD_PARTY_DIR ${ROOT}/third_party)
set (SETUP_DIR ${ROOT}/setup)
set (PYTHON_DIR ${ROOT}/graphlearn)
set (PYTHON_LIB ${PYTHON_DIR}/python/lib)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR})

# protobuf
find_package(Protobuf)
if(Protobuf_FOUND)
    set (PROTOBUF_INCLUDE ${Protobuf_INCLUDE_DIRS})
    get_filename_component(PROTOBUF_LIB ${Protobuf_LIBRARY} DIRECTORY)
    set (PROTOC ${Protobuf_PROTOC_EXECUTABLE})
    add_custom_target (protobuf_target)
    add_custom_command (TARGET protobuf_target
        COMMAND ${CMAKE_COMMAND} -E echo "protobuf found"
        VERBATIM
    )
else()
    set (PROTOBUF_DIR ${THIRD_PARTY_DIR}/protobuf)
    set (PROTOBUF_INCLUDE ${PROTOBUF_DIR}/build/include)
    set (PROTOBUF_LIB ${PROTOBUF_DIR}/build/lib)
    set (PROTOC ${PROTOBUF_DIR}/build/bin/protoc)
    add_custom_target (protobuf_target)
    add_custom_command (TARGET protobuf_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare protobuf library ..."
        COMMAND /bin/sh -c "if [ ! -d '${PROTOBUF_DIR}/build' ]; then cd '${PROTOBUF_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "protobuf done"
        VERBATIM
    )
endif()

# grpc
include(cmake/FindGRPC.cmake)
if(GRPC_LIBRARY)
    set (GRPC_INCLUDE ${GRPC_INCLUDE_DIR})
    get_filename_component(GRPC_LIB ${GRPC_LIBRARY} DIRECTORY)
    set (PROTOC_GRPC_PLUGIN ${GRPC_CPP_PLUGIN})
    add_custom_target (grpc_target)
    add_custom_command (TARGET grpc_target
        COMMAND ${CMAKE_COMMAND} -E echo "grpc found"
        VERBATIM
    )
else()
    set (GRPC_DIR ${THIRD_PARTY_DIR}/grpc)
    set (GRPC_INCLUDE ${GRPC_DIR}/build/include)
    set (GRPC_LIB ${GRPC_DIR}/build/lib)
    set (PROTOC_GRPC_PLUGIN ${GRPC_DIR}/build/bin/grpc_cpp_plugin)
    add_custom_target (grpc_target)
    add_custom_command (TARGET grpc_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare grpc library ..."
        COMMAND /bin/sh -c "if [ ! -d '${GRPC_DIR}/build' ]; then cd '${GRPC_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "grpc done"
        VERBATIM
    )
endif()
add_dependencies (grpc_target protobuf_target)

# glog
include(cmake/FindGlog.cmake)
if(GLOG_FOUND)
    set (GLOG_INCLUDE ${GLOG_INCLUDE_DIRS})
    get_filename_component(GLOG_LIB ${GLOG_LIBRARIES} DIRECTORY)
    add_custom_target (glog_target)
    add_custom_command (TARGET glog_target
        COMMAND ${CMAKE_COMMAND} -E echo "glog found"
        VERBATIM
    )
else()
    set (GLOG_DIR ${THIRD_PARTY_DIR}/glog)
    set (GLOG_INCLUDE ${GLOG_DIR}/build)
    set (GLOG_LIB ${GLOG_DIR}/build)
    add_custom_target (glog_target)
    add_custom_command (TARGET glog_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare glog library ..."
        COMMAND /bin/sh -c "if [ ! -d '${GLOG_DIR}/build' ]; then cd '${GLOG_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "glog done"
        VERBATIM
    )
endif()

# pybind11
find_package(pybind11)
if(pybind11_FOUND)
    set (PYBIND_DIR ${pybind11_INCLUDE_DIR}/..)
    set (PYBIND_INCLUDE ${pybind11_INCLUDE_DIR})
    add_custom_target (pybind_target)
    add_custom_command (TARGET pybind_target
        COMMAND ${CMAKE_COMMAND} -E echo "pybind11 found"
        VERBATIM
    )
else()
    set (PYBIND_DIR ${THIRD_PARTY_DIR}/pybind11)
    set (PYBIND_INCLUDE ${PYBIND_DIR}/pybind11/include/)
    add_custom_target (pybind_target)
    add_custom_command (TARGET pybind_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare pybind11 library ..."
        COMMAND /bin/sh -c "if [ ! -d '${PYBIND_DIR}/build' ]; then cd '${PYBIND_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "pybind11 done"
        VERBATIM
    )
endif()

# gtest
find_package(GTest)
if(GTEST_FOUND)
    set (GTEST_INCLUDE ${GTEST_INCLUDE_DIRS})
    get_filename_component(GTEST_LIB ${GTEST_MAIN_LIBRARIES} DIRECTORY)
    add_custom_target (gtest_target)
    add_custom_command (TARGET gtest_target
        COMMAND ${CMAKE_COMMAND} -E echo "gtest found"
        VERBATIM
    )
else()
    set (GTEST_DIR ${THIRD_PARTY_DIR}/googletest)
    set (GTEST_INCLUDE ${GTEST_DIR}/googletest/googletest/include)
    set (GTEST_LIB ${GTEST_DIR}/build/googlemock/gtest)
    add_custom_target (gtest_target)
    add_custom_command (TARGET gtest_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare gtest library ..."
        COMMAND /bin/sh -c "if [ ! -d '${GTEST_DIR}/build' ]; then cd '${GTEST_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "gtest done"
        VERBATIM
    )
endif()

# gflags
include(cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
    set (GFLAGS_INCLUDE ${GFLAGS_INCLUDE_DIR})
    get_filename_component(GFLAGS_LIB ${GFLAGS_LIBRARY} DIRECTORY)
    add_custom_target (gflags_target)
    add_custom_command (TARGET gflags_target
        COMMAND ${CMAKE_COMMAND} -E echo "gflags found"
        VERBATIM
    )
else()
    set (GFLAGS_DIR ${THIRD_PARTY_DIR}/gflags)
    set (GFLAGS_LIB ${GFLAGS_DIR}/build/lib)
    add_custom_target (gflags_target)
    add_custom_command (TARGET gflags_target
        COMMAND ${CMAKE_COMMAND} -E echo "prepare gflags library ..."
        COMMAND /bin/sh -c "if [ ! -d '${GFLAGS_DIR}/build' ]; then cd '${GFLAGS_DIR}'; ./build.sh; fi"
        COMMAND ${CMAKE_COMMAND} -E echo "gflags done"
        VERBATIM
    )
endif()

if(WITH_VINEYARD)
    find_package(vineyard)
    include_directories(SYSTEM ${VINEYARD_INCLUDE_DIRS})
    add_definitions(-DWITH_VINEYARD)
endif()

# proto
set (PROTO_DIR ${ROOT}/graphlearn/proto)
set (proto_generated_files
    ${PROTO_DIR}/service.pb.h 
    ${PROTO_DIR}/service.pb.cc
    ${PROTO_DIR}/service.grpc.pb.h 
    ${PROTO_DIR}/service.grpc.pb.cc)

add_custom_command (
    DEPENDS grpc_target
    OUTPUT ${proto_generated_files}
    COMMAND ${PROTOC} --cpp_out=. service.proto
    COMMAND ${PROTOC} --grpc_out=. service.proto
        --plugin=protoc-gen-grpc=${PROTOC_GRPC_PLUGIN}
    WORKING_DIRECTORY ${PROTO_DIR}
    VERBATIM
)

add_custom_target (proto_target
    DEPENDS ${proto_generated_files})

# common
set (COMMON_DIR ${ROOT}/graphlearn/common)
file (GLOB_RECURSE COMMON_FILES
    ${COMMON_DIR}/*.cc ${COMMON_DIR}/*.h)

# core
set (CORE_DIR ${ROOT}/graphlearn/core)
file (GLOB_RECURSE CORE_FILES
    ${CORE_DIR}/*.cc ${CORE_DIR}/*.h)

# include
set (INCLUDE_DIR ${ROOT}/graphlearn/include)
file (GLOB_RECURSE INCLUDE_FILES
    ${INCLUDE_DIR}/*.h)

# platform
set (PLATFORM_DIR ${ROOT}/graphlearn/platform)
file (GLOB_RECURSE PLATFORM_FILES
    ${PLATFORM_DIR}/*.cc ${PLATFORM_DIR}/*.h)

# service
set (SERVICE_DIR ${ROOT}/graphlearn/service)
file (GLOB_RECURSE SERVICE_FILES
    ${SERVICE_DIR}/*.cc ${SERVICE_DIR}/*.h)


if (GL_DEBUG)
    set (MODEFLAGS -DDEBUG -g)
else()
    set (MODEFLAGS -DNDEBUG -O2)
endif()

if (PROFILING)
    set (PROFILING_FLAG OPEN_PROFILING)
else()
    set (PROFILING_FLAG CLOSE_PROFILING)
endif()

set (GL_CXX_FLAGS ${MODEFLAGS} -std=c++14 -fPIC
    -pthread -mavx -msse4.2 -msse4.1 
    -D${PROFILING_FLAG} 
    -I. -I${ROOT} -I${BUILT_DIR} 
    -I${PROTOBUF_INCLUDE} 
    -I${GLOG_INCLUDE} 
    -I${GRPC_INCLUDE})

add_library (graphlearn_shared SHARED
    ${proto_generated_files}
    ${COMMON_FILES}
    ${CORE_FILES}
    ${INCLUDE_FILES}
    ${PLATFORM_FILES}
    ${SERVICE_FILES})

target_compile_options(graphlearn_shared 
    PRIVATE ${GL_CXX_FLAGS})

set (LINK_DIRS -L${GLOG_LIB} -L${ROOT} -L${PROTOBUF_LIB}
    -L${GRPC_LIB} -L${GFLAGS_LIB})

target_link_libraries (graphlearn_shared
    PUBLIC
    pthread
    glog
    protobuf
    gflags
    grpc++
    gpr
    upb
    ${LINK_DIRS})

if(WITH_VINEYARD)
    target_link_libraries(graphlearn_shared PUBLIC ${VINEYARD_LIBRARIES})
endif()

add_dependencies (graphlearn_shared 
    protobuf_target
    grpc_target
    glog_target
    pybind_target
    gtest_target
    gflags_target
    proto_target)

if (TESTING)
    function (gl_add_test name)
        cmake_parse_arguments (args
        "" "" "SOURCES" ${ARGN})

        add_executable (${name} ${args_SOURCES})
        target_include_directories(${name}
            PUBLIC ${GTEST_INCLUDE} 
            ${ROOT} ${PROTOBUF_INCLUDE} 
            ${GLOG_INCLUDE} ${GRPC_INCLUDE})
        target_link_libraries (${name}
            -L${GTEST_LIB} -L${LIB_DIR} -L${GRPC_LIB} -L${PROTOBUF_LIB}
            -L/lib64 -lgraphlearn_shared -lgtest -lprotobuf
            -lgtest_main -lstdc++ -lgrpc++ 
            -lgrpc -lgpr -lupb -lpthread)
        if(WITH_VINEYARD)
            target_link_libraries(${name} ${VINEYARD_LIBRARIES})
        endif()
        target_compile_options (${name}
            PRIVATE ${GL_CXX_FLAGS})
        add_dependencies (${name}
            gtest_target graphlearn_shared)
    endfunction()

    gl_add_test (closure_unittest
        SOURCES 
        graphlearn/common/base/test/closure_unittest.cpp)

    gl_add_test (status_unittest
        SOURCES 
        graphlearn/common/base/test/status_unittest.cpp
    )

    gl_add_test (atomic_unittest
        SOURCES 
        graphlearn/common/threading/atomic/test/atomic_unittest.cpp
    )

    gl_add_test (lockfree_queue_unittest
        SOURCES
        graphlearn/common/threading/lockfree/test/lockfree_queue_unittest.cpp
    )

    gl_add_test (lockfree_stack_unittest
        SOURCES
        graphlearn/common/threading/lockfree/test/lockfree_stack_unittest.cpp
    )

    gl_add_test (dynamic_worker_thread_pool_unittest
        SOURCES
        graphlearn/common/threading/runner/test/dynamic_worker_thread_pool_unittest.cpp
    )

    gl_add_test (cond_unittest
        SOURCES
        graphlearn/common/threading/sync/test/cond_unittest.cpp)

    gl_add_test (lock_unittest
        SOURCES
        graphlearn/common/threading/sync/test/lock_unittest.cpp)

    gl_add_test (waitable_event_unittest
        SOURCES
        graphlearn/common/threading/sync/test/waitable_event_unittest.cpp)

    gl_add_test (this_thread_unittest
        SOURCES
        graphlearn/common/threading/test/this_thread_unittest.cpp)

    gl_add_test (thread_unittest
        SOURCES
        graphlearn/common/threading/thread/test/thread_unittest.cpp)

    gl_add_test (graph_store_unittest
        SOURCES
        graphlearn/core/graph/test/graph_store_unittest.cpp)

    gl_add_test (node_storage_unittest
        SOURCES
        graphlearn/core/graph/storage/test/node_storage_unittest.cpp)

    gl_add_test (graph_storage_unittest
        SOURCES
        graphlearn/core/graph/storage/test/graph_storage_unittest.cpp)

    gl_add_test (data_slicer_unittest
        SOURCES
        graphlearn/core/io/test/data_slicer_unittest.cpp)

    gl_add_test (edge_loader_unittest
        SOURCES
        graphlearn/core/io/test/edge_loader_unittest.cpp)

    gl_add_test (node_loader_unittest
        SOURCES
        graphlearn/core/io/test/node_loader_unittest.cpp)

    gl_add_test (graph_op_unittest
        SOURCES
        graphlearn/core/operator/graph/test/graph_op_unittest.cpp)

    gl_add_test (sampler_unittest
        SOURCES
        graphlearn/core/operator/sampler/test/sampler_unittest.cpp)

    gl_add_test (negative_sampler_unittest
        SOURCES
        graphlearn/core/operator/sampler/test/negative_sampler_unittest.cpp)

    gl_add_test (aggregating_op_unittest
        SOURCES
        graphlearn/core/operator/aggregator/test/aggregating_op_unittest.cpp)

    gl_add_test (env_unittest
        SOURCES
        graphlearn/platform/test/env_unittest.cpp)

    gl_add_test (local_fs_unittest
        SOURCES
        graphlearn/platform/test/local_fs_unittest.cpp)

    gl_add_test (event_queue_unittest
        SOURCES
        graphlearn/service/test/event_queue_unittest.cpp)

    gl_add_test (tensor_unittest
        SOURCES
        graphlearn/service/test/tensor_unittest.cpp)

    gl_add_test (client_test
        SOURCES
        graphlearn/service/test/client_test.cpp)

    gl_add_test (server_test
        SOURCES
        graphlearn/service/test/server_test.cpp)

    gl_add_test (dist_in_memory_test
        SOURCES
        graphlearn/service/test/dist_in_memory_test.cpp)

    gl_add_test (graph_request_unittest
        SOURCES
        graphlearn/service/request/test/graph_request_unittest.cpp)

    gl_add_test (aggregating_request_unittest
        SOURCES
        graphlearn/service/request/test/aggregating_request_unittest.cpp)

    gl_add_test (naming_engine_unittest
        SOURCES
        graphlearn/service/dist/test/naming_engine_unittest.cpp)

    gl_add_test (coordinator_unittest
        SOURCES
        graphlearn/service/dist/test/coordinator_unittest.cpp)

    gl_add_test (channel_manager_unittest
        SOURCES
        graphlearn/service/dist/test/channel_manager_unittest.cpp)

    gl_add_test (service_unittest
        SOURCES
        graphlearn/service/dist/test/service_unittest.cpp)

    gl_add_test (service_with_hosts_unittest
        SOURCES
        graphlearn/service/dist/test/service_with_hosts_unittest.cpp)
endif()

execute_process (
    COMMAND /bin/sh -c "cat ${SETUP_DIR}/setup.py | grep '_VERSION =' | cut -d= -f2"
    OUTPUT_VARIABLE VERSION
    WORKING_DIRECTORY ${ROOT}
)

string(STRIP ${VERSION} VERSION)

# python
add_custom_target (python ALL)
add_dependencies (python pybind_target graphlearn_shared)
add_custom_command (TARGET python
    COMMAND /bin/sh -c "rm -rf build"
    COMMAND /bin/sh -c "rm -rf dist"
    COMMAND /bin/sh -c "rm -rf graphlearn.egg-info"
    COMMAND /bin/sh -c "mkdir -p ${PYTHON_LIB}"
    COMMAND /bin/sh -c "cp ${SETUP_DIR}/gl.__init__.py ${PYTHON_DIR}/__init__.py"
    COMMAND /bin/sh -c "echo '__version__ = ${VERSION}' >> ${PYTHON_DIR}/__init__.py"
    COMMAND /bin/sh -c "cp ${GRPC_LIB}/libgrpc++${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}/libgrpc++${CMAKE_SHARED_LIBRARY_SUFFIX} || true"
    COMMAND /bin/sh -c "cp ${GRPC_LIB}/libgrpc${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}/libgrpc${CMAKE_SHARED_LIBRARY_SUFFIX} || true"
    COMMAND /bin/sh -c "cp ${GRPC_LIB}/libgpr${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}/libgpr${CMAKE_SHARED_LIBRARY_SUFFIX} || true"
    COMMAND /bin/sh -c "cp ${GRPC_LIB}/libupb${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}/libupb${CMAKE_SHARED_LIBRARY_SUFFIX} || true"
    COMMAND /bin/sh -c "cp ${GLOG_LIB}/libglog${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}/libglog${CMAKE_SHARED_LIBRARY_SUFFIX} || true"
    COMMAND /bin/sh -c "cp ${LIB_DIR}/libgraphlearn_shared${CMAKE_SHARED_LIBRARY_SUFFIX} ${PYTHON_LIB}"
    COMMAND ${PYTHON_VERSION} ${SETUP_DIR}/setup.py bdist_wheel
    COMMAND /bin/sh -c "mkdir -p ${BIN_DIR}/ge_data/data"
    COMMAND /bin/sh -c "mkdir -p ${BIN_DIR}/ge_data/ckpt"
    COMMAND /bin/sh -c "rm -rf ${PYTHON_DIR}/__init__.py*"
    WORKING_DIRECTORY ${ROOT}
    VERBATIM
)
