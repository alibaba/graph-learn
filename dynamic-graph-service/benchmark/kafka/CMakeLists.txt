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

add_executable (kafka-driver
  main.cc
  driver.cc
  driver.h
  options.cc
  options.h)

target_link_libraries (kafka-driver
  PUBLIC
    CppKafka::cppkafka
    boost_program_options)

set_target_properties (kafka-driver
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GL_BUILT_BENCHMARK_DIR}/kafka)