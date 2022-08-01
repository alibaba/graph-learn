# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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
project (DynamicGraphService VERSION 1.0.0 LANGUAGES CXX)

## gcc version checks
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0.0")
  message (FATAL_ERROR "Building hiactor requires g++ version >= 8.x")
endif ()

## options
option (DEBUG "Enable debug mode" OFF)
option (TESTING "Enable testing targets" ON)
option (APPS "Enable apps targets" ON)
option (PROFILING "Enable profiling" OFF)
option (BENCHMARKING "Enable benchmarking" OFF)
option (USE_JEMALLOC "Enable memory allocation of jemalloc" ON)
option (ENABLE_GCOV "Enable gcov (debug mode only)" OFF)

if (ENABLE_GCOV AND (NOT DEBUG OR NOT TESTING))
  message (FATAL_ERROR "Code coverage is only supported in debug mode with testing")
endif ()

## project source paths
set (DGS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set (DGS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set (DGS_SRC_DIR ${DGS_ROOT}/src)
set (DGS_PYTHON_DIR ${DGS_ROOT}/python)
set (DGS_FBS_DIR ${DGS_ROOT}/fbs)
set (DGS_PROTO_DIR ${DGS_ROOT}/proto)
set (THIRD_PARTY_DIR ${DGS_ROOT}/../third_party)

## project target paths
set (DGS_BUILT_DIR ${DGS_ROOT}/built)
set (DGS_BUILT_LIB_DIR ${DGS_BUILT_DIR}/lib)
set (DGS_BUILT_BIN_DIR ${DGS_BUILT_DIR}/bin)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DGS_BUILT_BIN_DIR})
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DGS_BUILT_LIB_DIR})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DGS_BUILT_LIB_DIR})

## hiactor package
set (Hiactor_INSTALL_DIR ${THIRD_PARTY_DIR}/hiactor/build)
list (APPEND CMAKE_PREFIX_PATH ${Hiactor_INSTALL_DIR})
find_package (Hiactor CONFIG REQUIRED)

## hiactor codegen
include (${Hiactor_INSTALL_DIR}/bin/hiactor_codegen/ActorAutoGen.cmake)
hiactor_codegen (hiactor_codegen DGS_HIACTOR_AUTOGEN_FILES
  ACTOR_SOURCE_DIR ${DGS_SRC_DIR}/service
  HIACTOR_INCLUDE ${Hiactor_INSTALL_DIR}/include
  SYSTEM_INCLUDE ""
  USER_INCLUDE ${DGS_SRC_DIR})

## cppkafka package
set (CppKafka_INSTALL_DIR ${THIRD_PARTY_DIR}/cppkafka/build)
list (APPEND CMAKE_PREFIX_PATH ${CppKafka_INSTALL_DIR})
find_package (CppKafka CONFIG REQUIRED)

## flatbuffers package
set (Flatbuffers_INSTALL_DIR ${THIRD_PARTY_DIR}/flatbuffers/build)
list (APPEND CMAKE_PREFIX_PATH ${Flatbuffers_INSTALL_DIR})
find_package (Flatbuffers CONFIG REQUIRED)

## flatbuffers codegen
set (DGS_FBS_GEN_DIR ${DGS_SRC_DIR}/generated/fbs)
set (DGS_FBS_SRC_FILES
  ${DGS_FBS_DIR}/install_query_req.fbs
  ${DGS_FBS_DIR}/plan_node.fbs
  ${DGS_FBS_DIR}/query_plan.fbs
  ${DGS_FBS_DIR}/query_response.fbs
  ${DGS_FBS_DIR}/record.fbs
  ${DGS_FBS_DIR}/run_query_req.fbs
  ${DGS_FBS_DIR}/schema.fbs
  ${DGS_FBS_DIR}/stop_service.fbs
  ${DGS_FBS_DIR}/sync_meta.fbs
  ${DGS_FBS_DIR}/uninstall_query_req.fbs)

function (generate_fbs_headers target_name fbs_gen_headers)
  set (${fbs_gen_headers})
  foreach (FIL ${DGS_FBS_SRC_FILES})
    get_filename_component (FIL_WE ${FIL} NAME_WE)
    list (APPEND ${fbs_gen_headers} "${DGS_FBS_GEN_DIR}/${FIL_WE}_generated.h")
    add_custom_command (
      OUTPUT "${DGS_FBS_GEN_DIR}/${FIL_WE}_generated.h"
      DEPENDS flatbuffers::flatc ${FIL}
      COMMENT "-- Generating for flatbuffers file: ${FIL_WE}.fbs ..."
      COMMAND flatbuffers::flatc -o ${DGS_FBS_GEN_DIR} --gen-mutable --cpp ${FIL}
      WORKING_DIRECTORY
      VERBATIM)
  endforeach ()

  set_source_files_properties (${${fbs_gen_headers}} PROPERTIES GENERATED TRUE)
  set (${fbs_gen_headers} ${${fbs_gen_headers}} PARENT_SCOPE)

  add_custom_target (${target_name}
    DEPENDS ${${fbs_gen_headers}})
endfunction ()

generate_fbs_headers (fbs_codegen DGS_FBS_AUTOGEN_HEADERS)

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
set (DGS_GRPC_CPP_PLUGIN ${gRPC_INSTALL_DIR}/bin/grpc_cpp_plugin)
set (DGS_PROTO_GEN_DIR ${DGS_SRC_DIR}/generated/proto)
file (MAKE_DIRECTORY ${DGS_PROTO_GEN_DIR})
set (DGS_PROTO_SRC_FILES ${DGS_PROTO_DIR}/coordinator.proto)

function (generate_grpc_files target_name grpc_gen_files)
  set (${grpc_gen_files})
  foreach (FIL ${DGS_PROTO_SRC_FILES})
    get_filename_component (FIL_WE ${FIL} NAME_WE)
    set (GEN_FILES
      ${DGS_PROTO_GEN_DIR}/${FIL_WE}.pb.h
      ${DGS_PROTO_GEN_DIR}/${FIL_WE}.pb.cc
      ${DGS_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.h
      ${DGS_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.cc)
    list (APPEND ${grpc_gen_files} ${GEN_FILES})
    add_custom_command (
      OUTPUT ${GEN_FILES}
      DEPENDS protobuf::protoc ${DGS_GRPC_CPP_PLUGIN} ${FIL}
      COMMENT "-- Generating for proto file: ${FIL_WE}.proto ..."
      COMMAND protobuf::protoc -I ${DGS_PROTO_DIR} --cpp_out=${DGS_PROTO_GEN_DIR} ${FIL}
      COMMAND protobuf::protoc -I ${DGS_PROTO_DIR} --grpc_out=${DGS_PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${DGS_GRPC_CPP_PLUGIN} ${FIL}
      VERBATIM)
  endforeach ()

  set_source_files_properties (${${grpc_gen_files}} PROPERTIES GENERATED TRUE)
  set (${grpc_gen_files} ${${grpc_gen_files}} PARENT_SCOPE)

  add_custom_target (${target_name}
    DEPENDS ${${grpc_gen_files}})
endfunction ()

generate_grpc_files (grpc_codegen DGS_GRPC_AUTOGEN_FILES)

## rocksdb package
set (RocksDB_INSTALL_DIR ${THIRD_PARTY_DIR}/rocksdb/build)
list (APPEND CMAKE_PREFIX_PATH ${RocksDB_INSTALL_DIR})
find_package (RocksDB CONFIG REQUIRED)

## c-ares compatibility for grpc
if (c-ares_FOUND AND NOT (TARGET c-ares::cares))
  add_library (c-ares::cares UNKNOWN IMPORTED)

  set_target_properties (c-ares::cares
    PROPERTIES
    IMPORTED_LOCATION ${c-ares_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${c-ares_INCLUDE_DIRS})
endif ()

## project files
# common
set (DGS_COMMON_DIR ${DGS_SRC_DIR}/common)
file (GLOB_RECURSE DGS_COMMON_FILES ${DGS_COMMON_DIR}/*.cc)
# core
set (DGS_CORE_DIR ${DGS_SRC_DIR}/core)
file (GLOB_RECURSE DGS_CORE_FILES ${DGS_CORE_DIR}/*.cc)
# service
set (DGS_SERVICE_DIR ${DGS_SRC_DIR}/service)
file (GLOB_RECURSE DGS_SERVICE_FILES ${DGS_SERVICE_DIR}/*.cc)
list (FILTER DGS_SERVICE_FILES EXCLUDE REGEX ".*autogen\\.cc$")

## cxx flags
set (CXX_STD gnu++17)

if (DEBUG)
  add_definitions (-DDGS_DEBUG)
  set (DGS_MODE_FLAGS -DDEBUG -g)
  set (CMAKE_BUILD_TYPE Debug)
else ()
  set (DGS_MODE_FLAGS -DNDEBUG -O2)
endif ()

if (BENCHMARKING)
  add_definitions (-DDGS_BENCHMARK)
endif ()

if (PROFILING)
  set (DGS_PROFILING_FLAG -DOPEN_PROFILING)
else ()
  set (DGS_PROFILING_FLAG -DCLOSE_PROFILING)
endif ()

set (DGS_CXX_FLAGS
  ${DGS_MODE_FLAGS}
  ${DGS_PROFILING_FLAG}
  -std=${CXX_STD}
  -fPIC
  -fvisibility-inlines-hidden
  -pthread
  -mavx
  -msse4.2
  -msse4.1
  -Wno-attributes
  -Wno-deprecated-declarations
  -Werror=return-type)

if (ENABLE_GCOV)
  set (DGS_CXX_FLAGS ${DGS_CXX_FLAGS} -fprofile-arcs -ftest-coverage)
  set (CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif ()

## add library
add_library (dgs STATIC
  ${DGS_COMMON_FILES}
  ${DGS_CORE_FILES}
  ${DGS_SERVICE_FILES}
  ${DGS_HIACTOR_AUTOGEN_FILES}
  ${DGS_FBS_AUTOGEN_HEADERS}
  ${DGS_GRPC_AUTOGEN_FILES})

add_dependencies (hiactor_codegen
  fbs_codegen)
add_dependencies (dgs
  hiactor_codegen
  fbs_codegen
  grpc_codegen)

target_compile_options (dgs
  PRIVATE ${DGS_CXX_FLAGS})

target_include_directories (dgs
  PUBLIC ${DGS_SRC_DIR})

target_link_libraries (dgs
  PUBLIC
    Hiactor::hiactor
    CppKafka::cppkafka
    flatbuffers::flatbuffers
    glog::glog
    gRPC::grpc++
    RocksDB::rocksdb)

if (USE_JEMALLOC)
  pkg_search_module (JEMALLOC REQUIRED jemalloc)

  target_link_libraries (dgs
    PUBLIC ${JEMALLOC_LIBRARIES})
endif ()

if (ENABLE_GCOV)
  target_link_libraries (dgs
    PUBLIC gcov)
endif ()

## tests
if (TESTING)
  # googletest package
  set (GTest_INSTALL_DIR ${THIRD_PARTY_DIR}/googletest/build)
  list (APPEND CMAKE_PREFIX_PATH ${GTest_INSTALL_DIR})
  find_package (GTest CONFIG REQUIRED)

  # curl package
  find_package (CURL REQUIRED)

  add_custom_target (dgs_tests)
  function (add_dgs_tests)
    if (NOT ARGN)
      message (SEND_ERROR "Error: add_dgs_tests() called without any test files")
      return ()
    endif ()
    foreach (TEST_FILE ${ARGN})
      get_filename_component (TEST_NAME ${TEST_FILE} NAME_WE)
      add_executable (${TEST_NAME} ${TEST_FILE})
      add_dependencies (${TEST_NAME} dgs)
      target_compile_options (${TEST_NAME} PRIVATE ${DGS_CXX_FLAGS})
      target_link_libraries (${TEST_NAME}
        PUBLIC
          -Wl,--whole-archive
          dgs
          -Wl,--no-whole-archive
          GTest::gtest
          GTest::gtest_main
          CURL::libcurl)
      add_dependencies (dgs_tests ${TEST_NAME})
    endforeach ()
  endfunction ()

  file (GLOB_RECURSE DGS_TEST_FILES ${DGS_SRC_DIR}/*_unittest.cpp)
  add_dgs_tests (${DGS_TEST_FILES})

  if (ENABLE_GCOV)
    file (MAKE_DIRECTORY ${DGS_BUILD_DIR}/coverage/targets)
    set (DGS_COV_TARGET_FILES)
    foreach (ORIGIN_CC ${DGS_TEST_FILES})
      string (REGEX REPLACE "${DGS_ROOT}" "${DGS_BUILD_DIR}/CMakeFiles" OBJ_DIR ${ORIGIN_CC})
      string (REGEX REPLACE "(.*CMakeFiles)(.*)/test/(.*).cpp" "\\1/\\3.dir\\2/test" OBJ_DIR ${OBJ_DIR})
      get_filename_component (TEST_FILE ${ORIGIN_CC} NAME_WE)
      list (APPEND DGS_COV_TARGET_FILES
        ${DGS_BUILD_DIR}/coverage/targets/${TEST_FILE}.gcno
        ${DGS_BUILD_DIR}/coverage/targets/${TEST_FILE}.gcda)
      add_custom_command (
        OUTPUT
          ${DGS_BUILD_DIR}/coverage/targets/${TEST_FILE}.gcno
          ${DGS_BUILD_DIR}/coverage/targets/${TEST_FILE}.gcda
        COMMENT "-- Collecting code coverage targets of ${TEST_FILE}"
        COMMAND cp -f ${OBJ_DIR}/${TEST_FILE}.gcno ${DGS_BUILD_DIR}/coverage/targets/
        COMMAND cp -f ${OBJ_DIR}/${TEST_FILE}.gcda ${DGS_BUILD_DIR}/coverage/targets/
        WORKING_DIRECTORY ${DGS_BUILD_DIR})
    endforeach ()
    add_custom_target (code_coverage
      DEPENDS ${DGS_COV_TARGET_FILES})
  endif ()
endif ()

## apps
if (APPS)
  add_custom_target(dgs_apps)
  function (add_dgs_apps)
    if (NOT ARGN)
      message (SEND_ERROR "Error: add_dgs_apps() called without any test files")
      return ()
    endif ()
    foreach (APP_FILE ${ARGN})
      get_filename_component (APP_NAME ${APP_FILE} NAME_WE)
      add_executable (${APP_NAME} ${APP_FILE})
      add_dependencies (${APP_NAME} dgs)
      target_compile_options (${APP_NAME} PRIVATE ${DGS_CXX_FLAGS})
      target_link_libraries (${APP_NAME}
        PUBLIC
          -Wl,--whole-archive
          dgs
          -Wl,--no-whole-archive)
      add_dependencies (dgs_apps ${APP_NAME})
    endforeach ()
  endfunction ()

  add_dgs_apps (${DGS_SRC_DIR}/service/apps/naive_coordinator.cpp)
  add_dgs_apps (${DGS_SRC_DIR}/service/apps/service_main.cpp)
  add_dgs_apps (${DGS_SRC_DIR}/service/apps/query_install.cpp)
endif ()

## benchmarking
if (BENCHMARKING)
  add_subdirectory (benchmark)
endif ()

## packaging binary, dynamic libs and files
set (DGS_PACKAGE_DIR dgs_package)
set (DGS_PACKAGE_BIN_DIR ${DGS_PACKAGE_DIR}/bin)
set (DGS_PACKAGE_PYTHON_DIR ${DGS_PACKAGE_DIR}/python)
set (DGS_PACKAGE_FBS_DIR ${DGS_PACKAGE_DIR}/fbs)
set (DGS_PACKAGE_PROTO_DIR ${DGS_PACKAGE_DIR}/proto)
add_custom_target (package)
add_custom_command (TARGET package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DGS_PACKAGE_BIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:service_main> "${DGS_PACKAGE_BIN_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory "${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/__init__.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/barrier.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/checkpoint.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/coordinator.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/grpc_service.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/http_service.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"
  COMMAND /bin/bash -c "cp -f ${DGS_PYTHON_DIR}/coordinator/state_manager.py ${DGS_PACKAGE_PYTHON_DIR}/coordinator"

  COMMAND ${CMAKE_COMMAND} -E make_directory "${DGS_PACKAGE_FBS_DIR}"
  COMMAND /bin/bash -c "cp -f ${DGS_FBS_DIR}/*.fbs ${DGS_PACKAGE_FBS_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory "${DGS_PACKAGE_PROTO_DIR}"
  COMMAND /bin/bash -c "cp -f ${DGS_PROTO_DIR}/*.proto ${DGS_PACKAGE_PROTO_DIR}"

  COMMAND /bin/bash -c "tar zcvf dgs-built-${DynamicGraphService_VERSION}.tgz ${DGS_PACKAGE_DIR}"

  WORKING_DIRECTORY ${DGS_BUILD_DIR}
  VERBATIM)
