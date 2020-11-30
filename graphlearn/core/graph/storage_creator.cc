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

#include "graphlearn/core/graph/storage_creator.h"
#include "graphlearn/core/graph/storage/storage_mode.h"

namespace graphlearn {

#define CREATE(Type)                                 \
  if (io::IsVineyardStorageEnabled()) {              \
    return io::NewVineyard##Type##Storage(type, view_type);     \
  } else if (io::IsCompressedStorageEnabled()) {     \
    return io::NewCompressedMemory##Type##Storage(); \
  } else {                                           \
    return io::NewMemory##Type##Storage();           \
  }

io::GraphStorage* CreateGraphStorage(const std::string& type,
    const std::string& view_type) {
  CREATE(Graph)
}

io::NodeStorage* CreateNodeStorage(const std::string& type,
    const std::string& view_type) {
  CREATE(Node)
}

#undef CREATE

}  // namespace graphlearn
