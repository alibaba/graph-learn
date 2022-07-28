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

cmake_minimum_required (VERSION 3.10)
project (DataLoader VERSION 1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS_RELEASE "-O2")

set (DataLoader_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (DataLoader_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set (DGS_PROJECT_DIR ${DataLoader_SOURCE_DIR}/..)
set (ThirdParty_DIR ${DGS_PROJECT_DIR}/../third_party)

option (DataLoader_BUILD_SHARED "Build dataloader as a shared library." ON)
option (DataLoader_INSTALL "Install dataloader targets." ON)

if (DataLoader_BUILD_SHARED)
  set (DataLoader_LIBRARY_TYPE SHARED)
else ()
  set (DataLoader_LIBRARY_TYPE STATIC)
  set (CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

## ThirdParty
# CppKafka
list (APPEND CMAKE_PREFIX_PATH ${ThirdParty_DIR}/cppkafka/build)
find_package (CppKafka 0.3.1 REQUIRED)
# Flatbuffers
list (APPEND CMAKE_PREFIX_PATH ${ThirdParty_DIR}/flatbuffers/build)
find_package (Flatbuffers 1.12.0 REQUIRED)
# Boost
find_package (Boost COMPONENTS headers program_options thread)
# Curl
find_package (CURL REQUIRED)

## Generate flatbuffers header files
set (FBS_SRC_DIR
  ${DGS_PROJECT_DIR}/fbs)
set (FBS_SRC_FILES
  ${FBS_SRC_DIR}/record.fbs
  ${FBS_SRC_DIR}/schema.fbs)
set (FBS_GEN_DIR
  ${DataLoader_SOURCE_DIR}/include/dataloader/fbs)
file (MAKE_DIRECTORY ${FBS_GEN_DIR})
set (FBS_GEN_HEADER_FILES
  ${FBS_GEN_DIR}/record_generated.h
  ${FBS_GEN_DIR}/schema_generated.h)
add_custom_command (
  DEPENDS ${FBS_SRC_FILES} flatbuffers::flatc
  OUTPUT ${FBS_GEN_HEADER_FILES}
  COMMAND flatbuffers::flatc -o ${FBS_GEN_DIR} --gen-mutable --cpp ${FBS_SRC_FILES}
  VERBATIM)
add_custom_target (fbs_codegen DEPENDS ${FBS_GEN_HEADER_FILES})

## Build library
set (DataLoader_PROJECT_FILES
  include/dataloader/attribute.h
  include/dataloader/batch_builder.h
  include/dataloader/batch_producer.h
  include/dataloader/dataloader.h
  include/dataloader/group_producer.h
  include/dataloader/options.h
  include/dataloader/partitioner.h
  include/dataloader/schema.h
  include/dataloader/typedefs.h
  include/dataloader/utils.h
  src/batch_builder.cc
  src/batch_producer.cc
  src/dataloader.cc
  src/group_producer.cc
  src/schema.cc)

add_library (dataloader ${DataLoader_LIBRARY_TYPE}
  ${DataLoader_PROJECT_FILES}
  ${FBS_GEN_HEADER_FILES})

add_dependencies (dataloader
  fbs_codegen)

add_library (DataLoader::dataloader ALIAS dataloader)
set_target_properties (dataloader PROPERTIES CLEAN_DIRECT_OUTPUT 1)

target_include_directories (dataloader
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${DataLoader_SOURCE_DIR}/include>
  PRIVATE
    ${DataLoader_SOURCE_DIR}/src)

target_link_libraries (dataloader
  PUBLIC
    CppKafka::cppkafka
    flatbuffers::flatbuffers
    ${Boost_LIBRARIES}
    CURL::libcurl)

add_subdirectory (apps)

if (DataLoader_INSTALL)
  include (GNUInstallDirs)
  include (CMakePackageConfigHelpers)

  set (DataLoader_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/DataLoader)

  install (
    DIRECTORY ${DataLoader_SOURCE_DIR}/include
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install (
    TARGETS dataloader
    EXPORT dataloader-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install (
    EXPORT dataloader-export
    FILE DataLoaderTargets.cmake
    NAMESPACE DataLoader::
    DESTINATION ${DataLoader_CONFIG_INSTALL_DIR})

  configure_package_config_file (
    ${DataLoader_SOURCE_DIR}/cmake/DataLoaderConfig.cmake.in
    ${DataLoader_BINARY_DIR}/DataLoaderConfig.cmake
    INSTALL_DESTINATION ${DataLoader_CONFIG_INSTALL_DIR})

  write_basic_package_version_file (
    ${DataLoader_BINARY_DIR}/DataLoaderConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion)

  install (
    FILES
      ${DataLoader_BINARY_DIR}/DataLoaderConfig.cmake
      ${DataLoader_BINARY_DIR}/DataLoaderConfigVersion.cmake
    DESTINATION ${DataLoader_CONFIG_INSTALL_DIR}
    COMPONENT config)
endif ()