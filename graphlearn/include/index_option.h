/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_INCLUDE_INDEX_OPTION_H_
#define GRAPHLEARN_INCLUDE_INDEX_OPTION_H_

#include <cstdint>
#include <string>

namespace graphlearn {

struct IndexOption {
  std::string name;        // KNN or others
  std::string index_type;  // How to build index
  int32_t     dimension;
  int32_t     nlist;
  int32_t     nprobe;
  int32_t     m;

  IndexOption() {}

  IndexOption(const IndexOption& right) {
    name = right.name;
    index_type = right.index_type;
    dimension = right.dimension;
    nlist = right.nlist;
    nprobe = right.nprobe;
    m = right.m;
  }

  bool Empty() const {
    return name.empty();
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_INDEX_OPTION_H_
