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

#include "graphlearn/core/graph/storage/auto_indexing.h"

namespace graphlearn {
namespace io {

void AutoIndex::Add(IdType id) {
  IndexType index = converter_.size();
  converter_.insert({id, index});
}

IndexType AutoIndex::Get(IdType id) {
  auto it = converter_.find(id);
  if (it == converter_.end()) {
    return -1;
  } else {
    return it->second;
  }
}

}  // namespace io
}  // namespace graphlearn
