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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_AUTO_INDEXING_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_AUTO_INDEXING_H_

#include <cstdint>
#include <vector>
#include "graphlearn/core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class AutoIndex {
public:
  AutoIndex() = default;
  ~AutoIndex() = default;

  void Add(IdType id);
  IndexType Get(IdType);

private:
  MAP converter_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_AUTO_INDEXING_H_
