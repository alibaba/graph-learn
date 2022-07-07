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

add_app (file_data_loading_service
  file_data_loading_service.cc
  core/group_producer.cc
  core/group_producer.h
  core/loader.cc
  core/loader.h
  core/line_processer.cc
  core/line_processer.h)

# packaging file data loading service
set (FILE_DL_PACKAGE_DIR file_dl_package)
set (FILE_DL_PACKAGE_BIN_DIR ${FILE_DL_PACKAGE_DIR}/bin)
set (FILE_DL_PACKAGE_LIB_DIR ${FILE_DL_PACKAGE_DIR}/lib)
set (FILE_DL_PACKAGE_FBS_DIR ${FILE_DL_PACKAGE_DIR}/fbs)
add_custom_target (file_dl_package)
add_custom_command (TARGET file_dl_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${FILE_DL_PACKAGE_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:file_data_loading_service> "${FILE_DL_PACKAGE_BIN_DIR}"
  COMMAND /bin/bash -c "cp -f ${CMAKE_CURRENT_SOURCE_DIR}/scripts/dblp_preprossing.py ${FILE_DL_PACKAGE_BIN_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory ${FILE_DL_PACKAGE_LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dataloader> "${FILE_DL_PACKAGE_LIB_DIR}"

  COMMAND ${CMAKE_COMMAND} -E make_directory ${FILE_DL_PACKAGE_FBS_DIR}
  COMMAND /bin/bash -c "cp -f ${DGS_PROJECT_DIR}/fbs/*.fbs ${FILE_DL_PACKAGE_FBS_DIR}"

  COMMAND /bin/bash -c "tar zcvf file-dataloader-built.tgz ${FILE_DL_PACKAGE_DIR}"

  WORKING_DIRECTORY ${DataLoader_BINARY_DIR}
  VERBATIM)
