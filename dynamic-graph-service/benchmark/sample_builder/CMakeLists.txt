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

add_executable (sample_builder_test
  sample_builder_test.cc)
target_link_libraries (sample_builder_test
  PUBLIC
    -Wl,--whole-archive
    dgs
    -Wl,--no-whole-archive)
add_dependencies (sample_builder_test dgs)
set_target_properties (sample_builder_test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DGS_BUILT_BENCHMARK_DIR}/sample_builder)

add_executable (map_test map_test.cc)
set_target_properties (map_test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DGS_BUILT_BENCHMARK_DIR}/sample_builder)