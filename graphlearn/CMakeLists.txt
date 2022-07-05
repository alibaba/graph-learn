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

## options
option (TESTING
  "Enable testing"
  OFF)

option (DEBUG
  "Enable debug mode"
  OFF)

option (PROFILING
  "Enable profiling"
  OFF)

option (KNN
  "Enable knn"
  ON)

option (GPU
  "Enable gpu"
  OFF)

set (GL_CXX_DIALECT
  "c++11"
  CACHE
  STRING
  "Compile graphlearn according to the named C++ standard.")

## project source paths
set (GL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set (GL_SRC_DIR ${GL_ROOT}/src)
set (GL_PYTHON_DIR ${GL_ROOT}/python)
set (GL_SETUP_DIR ${GL_ROOT}/setup)
set (GL_PROTO_DIR ${GL_ROOT}/proto)
set (THIRD_PARTY_DIR ${GL_ROOT}/../third_party)

## project target paths
set (GL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set (GL_BUILT_DIR ${GL_ROOT}/built)
set (GL_BUILT_LIB_DIR ${GL_BUILT_DIR}/lib)
set (GL_BUILT_BIN_DIR ${GL_BUILT_DIR}/bin)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GL_BUILT_BIN_DIR})
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${GL_BUILT_LIB_DIR})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${GL_BUILT_LIB_DIR})

## Threads dependency
find_package (Threads REQUIRED)

## glog package
set (glog_INSTALL_DIR ${THIRD_PARTY_DIR}/glog/build)
list (APPEND CMAKE_PREFIX_PATH ${glog_INSTALL_DIR})
find_package (glog CONFIG REQUIRED)

## grpc package
set (gRPC_INSTALL_DIR ${THIRD_PARTY_DIR}/grpc/build)
list (APPEND CMAKE_PREFIX_PATH ${gRPC_INSTALL_DIR})
find_package (protobuf CONFIG REQUIRED)
find_package (gRPC CONFIG REQUIRED)

## grpc codegen
set (GL_GRPC_CPP_PLUGIN ${gRPC_INSTALL_DIR}/bin/grpc_cpp_plugin)
set (GL_PROTO_GEN_DIR ${GL_SRC_DIR}/generated/proto)
file (MAKE_DIRECTORY ${GL_PROTO_GEN_DIR})
set (GL_PROTO_SRC_FILES
  ${GL_PROTO_DIR}/dag.proto
  ${GL_PROTO_DIR}/request.proto
  ${GL_PROTO_DIR}/service.proto
  ${GL_PROTO_DIR}/tensor.proto)

function (generate_grpc_files target_name grpc_gen_files)
  set (${grpc_gen_files})
  foreach (FIL ${GL_PROTO_SRC_FILES})
    get_filename_component (FIL_WE ${FIL} NAME_WE)
    set (GEN_FILES
      ${GL_PROTO_GEN_DIR}/${FIL_WE}.pb.h
      ${GL_PROTO_GEN_DIR}/${FIL_WE}.pb.cc
      ${GL_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.h
      ${GL_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.cc)
    list (APPEND ${grpc_gen_files} ${GEN_FILES})
    add_custom_command (
      OUTPUT ${GEN_FILES}
      DEPENDS protobuf::protoc ${GL_GRPC_CPP_PLUGIN} ${FIL}
      COMMENT "Generating for proto file: ${FIL_WE}.proto ..."
      COMMAND protobuf::protoc -I ${GL_PROTO_DIR} --cpp_out=${GL_PROTO_GEN_DIR} ${FIL}
      COMMAND protobuf::protoc -I ${GL_PROTO_DIR} --grpc_out=${GL_PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${GL_GRPC_CPP_PLUGIN} ${FIL}
      VERBATIM)
  endforeach ()

  set_source_files_properties (${${grpc_gen_files}} PROPERTIES GENERATED TRUE)
  set (${grpc_gen_files} ${${grpc_gen_files}} PARENT_SCOPE)

  add_custom_target (${target_name}
    DEPENDS ${${grpc_gen_files}})
endfunction ()

generate_grpc_files (grpc_codegen GL_GRPC_AUTOGEN_FILES)

if (TESTING)
  # googletest package
  set (GTest_INSTALL_DIR ${THIRD_PARTY_DIR}/googletest/build)
  list (APPEND CMAKE_PREFIX_PATH ${GTest_INSTALL_DIR})
  find_package (GTest CONFIG REQUIRED)
endif ()

## project files
# common
set (GL_COMMON_DIR ${GL_SRC_DIR}/common)
file (GLOB_RECURSE GL_COMMON_FILES
  ${GL_COMMON_DIR}/*.cc
  ${GL_COMMON_DIR}/*.h)
# core
set (GL_CORE_DIR ${GL_SRC_DIR}/core)
file (GLOB_RECURSE GL_CORE_FILES
  ${GL_CORE_DIR}/*.cc
  ${GL_CORE_DIR}/*.h)
# include
set (GL_INCLUDE_DIR ${GL_SRC_DIR}/include)
file (GLOB_RECURSE GL_INCLUDE_FILES
  ${GL_INCLUDE_DIR}/*.h)
# platform
set (GL_PLATFORM_DIR ${GL_SRC_DIR}/platform)
file (GLOB_RECURSE GL_PLATFORM_FILES
  ${GL_PLATFORM_DIR}/*.cc
  ${GL_PLATFORM_DIR}/*.h)
# service
set (GL_SERVICE_DIR ${GL_SRC_DIR}/service)
file (GLOB_RECURSE GL_SERVICE_FILES
  ${GL_SERVICE_DIR}/*.cc
  ${GL_SERVICE_DIR}/*.h)
# contrib knn
set (GL_CONTRIB_DIR ${GL_SRC_DIR}/contrib)
if (KNN)
  add_subdirectory (${GL_CONTRIB_DIR}/knn)
endif ()

## cxx flags
if (DEBUG)
  set (GL_MODE_FLAGS -DDEBUG -g)
  set (CMAKE_BUILD_TYPE Debug)
else ()
  set (GL_MODE_FLAGS -DNDEBUG -O2)
endif ()

## knn flag
if (KNN)
  set (KNN_FLAG OPEN_KNN)
else ()
  set (KNN_FLAG CLOSE_KNN)
endif ()

## profiling flag
if (PROFILING)
  set (PROFILING_FLAG OPEN_PROFILING)
else()
  set (PROFILING_FLAG CLOSE_PROFILING)
endif()

set (GL_CXX_FLAGS
  ${GL_MODE_FLAGS}
  -std=${GL_CXX_DIALECT}
  -D${KNN_FLAG}
  -D${PROFILING_FLAG}
  -fPIC
  -fvisibility-inlines-hidden
  -pthread
  -mavx
  -msse4.2
  -msse4.1
  # fixme: check format security
  -Wno-format-security)

## add library
add_library (graphlearn_shared SHARED
  ${GL_COMMON_FILES}
  ${GL_CORE_FILES}
  ${GL_INCLUDE_FILES}
  ${GL_PLATFORM_FILES}
  ${GL_SERVICE_FILES}
  ${GL_GRPC_AUTOGEN_FILES})

add_dependencies (graphlearn_shared
  grpc_codegen)

target_compile_options (graphlearn_shared
  PRIVATE ${GL_CXX_FLAGS})

target_include_directories (graphlearn_shared
  PUBLIC
    ${GL_SRC_DIR}
    ${THIRD_PARTY_DIR})

target_link_libraries (graphlearn_shared
  PUBLIC
    glog::glog
    gRPC::grpc++)

if (KNN)
  target_link_libraries (graphlearn_shared
    PUBLIC knn_shared)
endif ()

## tests
if (TESTING)
  add_custom_target (gl_tests)
  function (add_gl_tests)
    if (NOT ARGN)
      message (SEND_ERROR "Error: add_gl_tests() called without any test files")
      return ()
    endif ()
    foreach (TEST_FILE ${ARGN})
      get_filename_component (TEST_NAME ${TEST_FILE} NAME_WE)
      add_executable (${TEST_NAME} ${TEST_FILE})
      add_dependencies (${TEST_NAME} graphlearn_shared)
      target_compile_options (${TEST_NAME} PRIVATE ${GL_CXX_FLAGS})
      target_link_libraries (${TEST_NAME}
        PUBLIC
          graphlearn_shared
          GTest::gtest
          GTest::gtest_main)
      add_dependencies (gl_tests ${TEST_NAME})
    endforeach ()
  endfunction ()

  file (GLOB_RECURSE GL_TEST_FILES
    ${GL_COMMON_DIR}/*.cpp
    ${GL_CORE_DIR}/*.cpp
    ${GL_INCLUDE_DIR}/*.cpp
    ${GL_PLATFORM_DIR}/*.cpp
    ${GL_SERVICE_DIR}/*.cpp)
  add_gl_tests (${GL_TEST_FILES})

  if (KNN)
    file (GLOB_RECURSE KNN_TEST_FILES
      ${GL_CONTRIB_DIR}/knn/*.cpp)
    add_gl_tests (${KNN_TEST_FILES})
  endif ()
endif()

# python
execute_process (
  COMMAND /bin/sh -c "grep '_VERSION = ' ${GL_SETUP_DIR}/setup.py | cut -d= -f2"
  OUTPUT_VARIABLE VERSION
)
execute_process (
  COMMAND /bin/sh -c "git rev-parse --abbrev-ref HEAD"
  OUTPUT_VARIABLE GIT_BRANCH
)
execute_process (
  COMMAND /bin/sh -c "git rev-parse --short HEAD"
  OUTPUT_VARIABLE GIT_VERSION
)

add_custom_target (python)
add_custom_command (TARGET python
  COMMAND ${CMAKE_COMMAND} -E remove_directory dist
  COMMAND ${CMAKE_COMMAND} -E remove_directory graphlearn.egg-info
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GL_PYTHON_LIB_DIR}"
  COMMAND /bin/sh -c "cp -f ${GL_SETUP_DIR}/gl.__init__.py ${GL_PYTHON_DIR}/__init__.py"
  COMMAND /bin/sh -c "echo '__version__ = ${VERSION}' > ${GL_PYTHON_DIR}/__init__.py"
  COMMAND /bin/sh -c "echo '__git_version__ = '${GIT_BRANCH}-${GIT_VERSION}'' > ${GL_PYTHON_DIR}/__init__.py"
  COMMAND OPEN_KNN=${KNN_FLAG} python ${GL_SETUP_DIR}/setup.py bdist_wheel
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GL_BUILT_BIN_DIR}/ge_data/data"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GL_BUILT_BIN_DIR}/ge_data/ckpt"
  WORKING_DIRECTORY ${GL_ROOT}
  VERBATIM)
