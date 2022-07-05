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

set (KNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set (CUDA_PATH /usr/local/cuda-10.0)

execute_process (
  COMMAND /bin/sh -c "expr `g++ -dumpversion | cut -f1 -d.` \\>= 5"
  OUTPUT_VARIABLE GXX_VERSION_GTE_5
)

## prepare faiss
set (FAISS_BUILT_DIR ${KNN_ROOT}/faiss)
file (MAKE_DIRECTORY ${FAISS_BUILT_DIR})
set (FAISS_INCLUDE_DIR ${FAISS_BUILT_DIR}/include)
set (FAISS_LIB_DIR ${FAISS_BUILT_DIR}/lib)

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
  COMMAND /bin/sh -c "wget -O ${FAISS_PKG} ${FAISS_RESOURCE}"
  COMMAND /bin/sh -c "tar -zxf ${FAISS_PKG}"
  WORKING_DIRECTORY ${FAISS_BUILT_DIR}
  VERBATIM)

add_custom_target (faiss_download
  DEPENDS ${FAISS_DEPENDENT_FILES})

## gpu flag
if (GPU)
  set (GPU_FLAG OPEN_GPU)
else ()
  set (GPU_FLAG CLOSE_GPU)
endif ()

## knn cxx flags
set (KNN_CXX_FLAGS
  ${GL_MODE_FLAGS}
  -std=${GL_CXX_DIALECT}
  -D${GPU_FLAG}
  -fPIC
  -pthread
  -mavx
  -msse4.2
  -msse4.1)

## knn source files
file (GLOB_RECURSE KNN_SRC_FILES
  ${KNN_ROOT}/*.cc
  ${KNN_ROOT}/*.h)

## add library
add_library (knn_shared SHARED
  ${KNN_SRC_FILES})

add_dependencies (knn_shared
  faiss_download)

target_compile_options (knn_shared
  PRIVATE ${KNN_CXX_FLAGS})

target_include_directories (knn_shared
  PUBLIC
    ${GL_SRC_DIR}
    ${FAISS_INCLUDE_DIR})

target_link_directories (knn_shared
  PUBLIC ${FAISS_LIB_DIR})

target_link_libraries (knn_shared
  PUBLIC
    glog::glog
    gRPC::grpc++
    faiss_static)

target_link_options (knn_shared
  PUBLIC -lopenblas -lgomp -lm -lquadmath -lgfortran)

if (GPU)
  target_include_directories (knn_shared
    PUBLIC ${CUDA_PATH}/include)

  target_link_directories (knn_shared
    PUBLIC ${CUDA_PATH}/lib64)

  target_link_libraries (knn_shared
    PUBLIC cudart cublas)
endif ()

## knn tests
if (TESTING)
  add_custom_target (knn_tests)
  function (add_knn_tests)
    if (NOT ARGN)
      message (SEND_ERROR "Error: add_knn_tests() called without any test files")
      return ()
    endif ()
    foreach (TEST_FILE ${ARGN})
      get_filename_component (TEST_NAME ${TEST_FILE} NAME_WE)
      add_executable (${TEST_NAME} ${TEST_FILE})
      add_dependencies (${TEST_NAME} knn_shared)
      target_compile_options (${TEST_NAME} PRIVATE ${KNN_CXX_FLAGS})
      target_link_libraries (${TEST_NAME}
        PUBLIC
          knn_shared
          GTest::gtest
          GTest::gtest_main)
      add_dependencies (knn_tests ${TEST_NAME})
    endforeach ()
  endfunction ()

  file (GLOB_RECURSE KNN_TEST_FILES ${KNN_ROOT}/*.cpp)
  add_knn_tests (${KNN_TEST_FILES})
endif ()