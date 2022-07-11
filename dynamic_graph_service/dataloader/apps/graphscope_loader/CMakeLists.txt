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

## Graphscope-Store SDK
set (GS_SDK_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/maxgraph/native)
find_package (Lgraph 0.1.0 REQUIRED PATHS ${GS_SDK_ROOT_DIR})

add_executable (graphscope_data_loading_service
  graphscope_data_loading_service.cc
  core/bulk_loader.cc
  core/bulk_loader.h
  core/checkpoint_parser.cc
  core/checkpoint_parser.h
  core/gs_batch_builder.cc
  core/gs_batch_builder.h
  core/gs_option.cc
  core/gs_option.h
  core/gs_service.cc
  core/gs_service.h
  core/log_poller.cc
  core/log_poller.h)
add_dependencies (graphscope_data_loading_service
  dataloader)
target_link_libraries (graphscope_data_loading_service
  PUBLIC dataloader Lgraph::lgraph)

add_executable (graphscope_record_generator
  graphscope_record_generator.cc)
target_link_libraries (graphscope_record_generator
  PUBLIC Lgraph::lgraph Boost::program_options)

# packaging graphscope data loading service
set (GS_DL_PACKAGE_DIR gs_dl_package)
set (GS_DL_PACKAGE_BIN_DIR ${GS_DL_PACKAGE_DIR}/bin)
set (GS_DL_PACKAGE_LIB_DIR ${GS_DL_PACKAGE_DIR}/lib)
set (GS_DL_PACKAGE_FBS_DIR ${GS_DL_PACKAGE_DIR}/fbs)
add_custom_target (gs_dl_package)
add_custom_command (TARGET gs_dl_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GS_DL_PACKAGE_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:graphscope_data_loading_service> "${GS_DL_PACKAGE_BIN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:graphscope_record_generator> "${GS_DL_PACKAGE_BIN_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory ${GS_DL_PACKAGE_LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dataloader> "${GS_DL_PACKAGE_LIB_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Lgraph::lgraph> "${GS_DL_PACKAGE_LIB_DIR}"
  COMMAND /bin/bash -c "cp -f ${GS_SDK_ROOT_DIR}/lib*/libmaxgraph_ffi.* ${GS_DL_PACKAGE_LIB_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory ${GS_DL_PACKAGE_FBS_DIR}
  COMMAND /bin/bash -c "cp -f ${DGS_PROJECT_DIR}/fbs/*.fbs ${GS_DL_PACKAGE_FBS_DIR}"

  COMMAND /bin/bash -c "tar zcvf gs-dataloader-built.tgz ${GS_DL_PACKAGE_DIR}"

  WORKING_DIRECTORY ${DataLoader_BINARY_DIR}
  VERBATIM)

