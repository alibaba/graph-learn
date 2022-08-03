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

cmake_minimum_required (VERSION 3.13)

project (GraphLearn VERSION 1.0 LANGUAGES CXX)

## options
option (TESTING
  "Enable testing"
  ON)

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

option (WITH_VINEYARD
  "Enable vineyard"
  OFF)

option (VINEYARD_USE_OID
  "Use OID when work with vineyard graphs"
  ON)

# FIXME: change to OFF
option (WITH_HIACTOR
  "Enable hiactor engine"
  ON)

set (GL_CXX_DIALECT
  "c++11"
  CACHE
  STRING
  "Compile graphlearn according to the named C++ standard.")

set (GL_PYTHON_BIN
  "python3"
  CACHE
  STRING
  "Python binary to use.")

## project source paths
set (GL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set (GL_SRC_DIR ${GL_ROOT}/src)
set (GL_PYTHON_DIR ${GL_ROOT})
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

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Threads dependency
find_package (Threads REQUIRED)

## glog package
include (${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGlog.cmake)
if (GLOG_FOUND)
  add_library (glog::glog UNKNOWN IMPORTED)
  include (${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGFlags.cmake)
  set_target_properties (glog::glog PROPERTIES IMPORTED_LOCATION "${GLOG_LIBRARY}")
  set_target_properties (glog::glog PROPERTIES INTERFACE_LINK_LIBRARIES "${GFLAGS_LIBRARIES}")
  set_target_properties (glog::glog PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${GFLAGS_INCLUDE_DIRS}>;$<BUILD_INTERFACE:${GLOG_INCLUDE_DIRS}>")
else ()
  set (glog_INSTALL_DIR ${THIRD_PARTY_DIR}/glog/build)
  list (APPEND CMAKE_PREFIX_PATH ${glog_INSTALL_DIR})
  find_package (glog CONFIG REQUIRED)
endif ()

## hiactor package
if (APPLE)
  # hiactor is not buildable on MacOS
  set (WITH_HIACTOR OFF)
endif ()
set (Hiactor_INSTALL_DIR ${THIRD_PARTY_DIR}/hiactor/build)
if (WITH_HIACTOR)
  message("-- GraphLearn is built with hiactor engine.")
  list (APPEND CMAKE_PREFIX_PATH ${Hiactor_INSTALL_DIR})
  find_package (Hiactor CONFIG REQUIRED)

  ## c-ares compatibility with grpc
  if (c-ares_FOUND AND NOT (TARGET c-ares::cares))
    add_library (c-ares::cares UNKNOWN IMPORTED)

    set_target_properties (c-ares::cares
      PROPERTIES
        IMPORTED_LOCATION ${c-ares_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${c-ares_INCLUDE_DIRS})
  endif ()
endif ()

## protobuf and gRPC package
find_package (Protobuf QUIET)
if (NOT Protobuf_FOUND)
  set (gRPC_INSTALL_DIR ${THIRD_PARTY_DIR}/grpc/build)
  list (APPEND CMAKE_PREFIX_PATH ${gRPC_INSTALL_DIR})
  find_package (protobuf CONFIG REQUIRED)
endif ()

## grpc install by homebrew requires openssl
if (APPLE)
  # If we're on OS X check for Homebrew's copy of OpenSSL instead of Apple's
  if (NOT OpenSSL_DIR)
      find_program(HOMEBREW brew)
      if (HOMEBREW STREQUAL "HOMEBREW-NOTFOUND")
          message(WARNING "Homebrew not found: not using Homebrew's OpenSSL")
          if (NOT OPENSSL_ROOT_DIR)
              message(WARNING "Use -DOPENSSL_ROOT_DIR for non-Apple OpenSSL")
          endif()
      else()
          execute_process(COMMAND brew --prefix openssl
              OUTPUT_VARIABLE OPENSSL_ROOT_DIR
              OUTPUT_STRIP_TRAILING_WHITESPACE)
      endif()
  endif()
  find_package(OpenSSL ${ARGN})
endif()

macro (get_target_import_location var target)
  if (TARGET ${target})
    get_target_property(${var} ${target} IMPORTED_LOCATION)
    if ("${${var}}" STREQUAL "${var}-NOTFOUND")
      get_target_property(${var} ${target} IMPORTED_LOCATION_RELEASE)
    endif ()
  endif ()
endmacro ()

find_package (gRPC CONFIG QUIET)
if (gRPC_FOUND)
  if (TARGET gRPC::grpc_cpp_plugin)
    get_target_import_location(GL_GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin)
  else ()
    set (GL_GRPC_CPP_PLUGIN ${gRPC_INSTALL_DIR}/bin/grpc_cpp_plugin)
  endif ()
else ()
  include (${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGRPC.cmake)
  set_target_properties(gRPC::grpc++ PROPERTIES INTERFACE_LINK_LIBRARIES "${Protobuf_LIBRARIES}")
  get_target_import_location(GL_GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin)
endif ()

## grpc codegen
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
      DEPENDS protobuf::protoc gRPC::grpc_cpp_plugin ${FIL}
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

## project files
# common
set (GL_COMMON_DIR ${GL_SRC_DIR}/common)
file (GLOB_RECURSE GL_COMMON_FILES ${GL_COMMON_DIR}/*.cc)
# core
set (GL_CORE_DIR ${GL_SRC_DIR}/core)
file (GLOB_RECURSE GL_CORE_FILES ${GL_CORE_DIR}/*.cc)
# platform
set (GL_PLATFORM_DIR ${GL_SRC_DIR}/platform)
file (GLOB_RECURSE GL_PLATFORM_FILES ${GL_PLATFORM_DIR}/*.cc)
# service
set (GL_SERVICE_DIR ${GL_SRC_DIR}/service)
file (GLOB_RECURSE GL_SERVICE_FILES ${GL_SERVICE_DIR}/*.cc)

# actor
if (WITH_HIACTOR)
  set (GL_ACTOR_DIR ${GL_SRC_DIR}/actor)
  # actor codegen
  include (${Hiactor_INSTALL_DIR}/bin/hiactor_codegen/ActorAutoGen.cmake)
  hiactor_codegen (actor_codegen GL_ACTOR_AUTOGEN_FILES
    ACTOR_SOURCE_DIR ${GL_ACTOR_DIR}
    HIACTOR_INCLUDE ${Hiactor_INSTALL_DIR}/include
    SYSTEM_INCLUDE ""
    USER_INCLUDE ${GL_SRC_DIR})

  file (GLOB_RECURSE GL_ACTOR_SRC_FILES ${GL_ACTOR_DIR}/*.cc)
  list (FILTER GL_ACTOR_SRC_FILES EXCLUDE REGEX ".*autogen\\.cc$")

  set (GL_ACTOR_FILES
    ${GL_ACTOR_SRC_FILES}
    ${GL_ACTOR_AUTOGEN_FILES})
else ()
  set (GL_ACTOR_FILES)
endif ()

# contrib knn
if (APPLE)
  # knn module is not buildable on MacOS
  set(KNN OFF)
endif ()
set (GL_KNN_DIR ${GL_SRC_DIR}/contrib/knn)
set (FAISS_BUILT_DIR ${GL_KNN_DIR}/faiss)
set (FAISS_INCLUDE_DIR ${FAISS_BUILT_DIR}/include)
set (FAISS_LIB_DIR ${FAISS_BUILT_DIR}/lib)
if (KNN)
  execute_process (
    COMMAND /bin/sh -c "expr `g++ -dumpversion | cut -f1 -d.` \\>= 5"
    OUTPUT_VARIABLE GXX_VERSION_GTE_5
  )

  # prepare faiss
  file (MAKE_DIRECTORY ${FAISS_BUILT_DIR})
  set (FAISS_DEPENDENT_FILES
    ${FAISS_INCLUDE_DIR}/faiss/IndexFlat.h
    ${FAISS_INCLUDE_DIR}/faiss/IndexIVFFlat.h)

  if (${GXX_VERSION_GTE_5} MATCHES "1")
    set (FAISS_PKG faiss.tar.gz)
  else ()
    set (FAISS_PKG faiss_gcc4.8.5.tar.gz)
  endif ()
  set (FAISS_RESOURCE https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/${FAISS_PKG})

  add_custom_command (
    OUTPUT ${FAISS_DEPENDENT_FILES}
    COMMENT "downloading faiss package: ${FAISS_PKG} ..."
    COMMAND /bin/sh -c "wget -q -O ${FAISS_PKG} ${FAISS_RESOURCE}"
    COMMAND /bin/sh -c "tar -zxf ${FAISS_PKG}"
    WORKING_DIRECTORY ${FAISS_BUILT_DIR}
    VERBATIM)

  add_custom_target (faiss_download
    DEPENDS ${FAISS_DEPENDENT_FILES})

  # knn source files
  file (GLOB_RECURSE GL_KNN_FILES
    ${GL_KNN_DIR}/*.cc
    ${GL_KNN_DIR}/*.h)
else ()
  set (GL_KNN_FILES)
endif ()

## contrib vineyard
if (WITH_VINEYARD)
  message(INFO "Graph-learn is built with vineyard support.")
  find_package (vineyard)
  include_directories (SYSTEM ${VINEYARD_INCLUDE_DIRS})
  add_definitions (-DWITH_VINEYARD)
  if (VINEYARD_USE_OID)
    add_definitions (-DVINEYARD_USE_OID)
  endif ()
endif ()

## cxx flags
if (DEBUG)
  set (GL_MODE_FLAGS -DDEBUG -g)
  set (CMAKE_BUILD_TYPE Debug)
else ()
  set (GL_MODE_FLAGS -DNDEBUG -O2)
endif ()

## actor engine flag
if (WITH_HIACTOR)
  set (ACTOR_FLAG OPEN)
else ()
  set (ACTOR_FLAG CLOSE)
endif ()

## knn flag
if (KNN)
  set (KNN_FLAG OPEN)
else ()
  set (KNN_FLAG CLOSE)
endif ()

## gpu flag
if (GPU)
  set (GPU_FLAG OPEN)
else ()
  set (GPU_FLAG CLOSE)
endif ()

## profiling flag
if (PROFILING)
  set (PROFILING_FLAG OPEN)
else()
  set (PROFILING_FLAG CLOSE)
endif()

set (GL_CXX_FLAGS
  ${GL_MODE_FLAGS}
  -std=${GL_CXX_DIALECT}
  -D${ACTOR_FLAG}_ACTOR_ENGINE
  -D${KNN_FLAG}_KNN
  -D${GPU_FLAG}_GPU
  -D${PROFILING_FLAG}_PROFILING
  -fPIC
  -fvisibility-inlines-hidden
  -pthread
  -Wno-deprecated-declarations
  -Wno-format-security)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set (GL_CXX_FLAGS ${GL_CXX_FLAGS}
    -mavx
    -msse4.2
    -msse4.1)
endif ()

if (WITH_HIACTOR)
  set (GL_CXX_FLAGS ${GL_CXX_FLAGS}
    -Wno-attributes)
endif ()

## add library
add_library (graphlearn_shared SHARED
  ${GL_COMMON_FILES}
  ${GL_CORE_FILES}
  ${GL_PLATFORM_FILES}
  ${GL_SERVICE_FILES}
  ${GL_ACTOR_FILES}
  ${GL_KNN_FILES}
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

if (WITH_HIACTOR)
  target_link_libraries(graphlearn_shared
    PUBLIC
      -Wl,--whole-archive
      Hiactor::hiactor
      -Wl,--no-whole-archive)
endif ()

if (KNN)
  add_dependencies (graphlearn_shared
    faiss_download)

  target_include_directories (graphlearn_shared
    PUBLIC ${FAISS_INCLUDE_DIR})

  target_link_directories (graphlearn_shared
    PUBLIC ${FAISS_LIB_DIR})

  target_link_libraries (graphlearn_shared
    PUBLIC faiss_static openblas gomp)

  target_link_options (graphlearn_shared
    PUBLIC -lm -lquadmath -lgfortran)
endif ()

if (WITH_VINEYARD)
  target_include_directories (graphlearn_shared PUBLIC ${VINEYARD_INCLUDE_DIRS})
  target_link_libraries (graphlearn_shared PUBLIC ${VINEYARD_LIBRARIES})
endif ()

if (GPU)
  set (CUDA_PATH /usr/local/cuda-10.0)

  target_include_directories (graphlearn_shared
    PUBLIC ${CUDA_PATH}/include)

  target_link_directories (graphlearn_shared
    PUBLIC ${CUDA_PATH}/lib64)

  target_link_libraries (graphlearn_shared
    PUBLIC cudart cublas)
endif ()

## tests
if (TESTING)
  # googletest package
  include (${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGTest.cmake)
  if (GTEST_FOUND)
    add_library (GTest::gtest UNKNOWN IMPORTED)
    set_target_properties (GTest::gtest PROPERTIES IMPORTED_LOCATION "${GTEST_LIBRARY}")
    set_target_properties (GTest::gtest PROPERTIES INTERFACE_LINK_LIBRARIES "${GTEST_LIBRARIES}")
    set_target_properties (GTest::gtest PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>")
    add_library (GTest::gtest_main UNKNOWN IMPORTED)
    set_target_properties (GTest::gtest_main PROPERTIES IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}")
    set_target_properties (GTest::gtest_main PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>")
  else ()
    set (GTest_INSTALL_DIR ${THIRD_PARTY_DIR}/googletest/build)
    list (APPEND CMAKE_PREFIX_PATH ${GTest_INSTALL_DIR})
    find_package (GTest CONFIG REQUIRED)
  endif ()

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
    ${GL_PLATFORM_DIR}/*.cpp
    ${GL_SERVICE_DIR}/*.cpp)
  add_gl_tests (${GL_TEST_FILES})

  if (KNN)
    file (GLOB_RECURSE KNN_TEST_FILES
      ${GL_KNN_DIR}/*.cpp)
    add_gl_tests (${KNN_TEST_FILES})
  endif ()

  if (WITH_VINEYARD)
    target_compile_options(vineyard_storage_unittest PRIVATE "-std=c++14")
  endif ()
endif()

# python
execute_process (
  COMMAND /bin/sh -c "grep '_VERSION = ' ${GL_SETUP_DIR}/setup.py | cut -d= -f2"
  OUTPUT_VARIABLE VERSION
)
string (STRIP ${VERSION} VERSION)

execute_process (
  COMMAND /bin/sh -c "git rev-parse --abbrev-ref HEAD"
  OUTPUT_VARIABLE GIT_BRANCH
)
# FIXME: uncomment
#string (STRIP ${GIT_BRANCH} GIT_BRANCH)

execute_process (
  COMMAND /bin/sh -c "git rev-parse --short HEAD"
  OUTPUT_VARIABLE GIT_VERSION
)
# FIXME: uncomment
#string (STRIP ${GIT_VERSION} GIT_VERSION)

set (GL_PYTHON_LIB_DIR ${GL_PYTHON_DIR}/python/lib)
add_custom_target (python)
add_dependencies (python
  graphlearn_shared)
add_custom_command (TARGET python
  COMMAND ${CMAKE_COMMAND} -E remove_directory dist
  COMMAND ${CMAKE_COMMAND} -E remove_directory graphlearn.egg-info
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GL_PYTHON_LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:graphlearn_shared> ${GL_PYTHON_LIB_DIR}
  COMMAND cp -f ${GL_SETUP_DIR}/gl.__init__.py ${GL_PYTHON_DIR}/__init__.py
  COMMAND echo "__version__ = ${VERSION}" >> ${GL_PYTHON_DIR}/__init__.py
  COMMAND echo "__git_version__ = '${GIT_BRANCH}-${GIT_VERSION}'" >> ${GL_PYTHON_DIR}/__init__.py
  COMMAND OPEN_KNN=${KNN_FLAG} ${GL_PYTHON_BIN} ${GL_SETUP_DIR}/setup.py bdist_wheel
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GL_BUILT_BIN_DIR}/ge_data/data"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GL_BUILT_BIN_DIR}/ge_data/ckpt"
  COMMAND rm -f ${GL_PYTHON_DIR}/__init__.py
  WORKING_DIRECTORY ${GL_ROOT}
  VERBATIM)

# install the graphlearn_shared library
install(TARGETS graphlearn_shared
        EXPORT graphlearn-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
