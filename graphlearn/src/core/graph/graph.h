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

#ifndef GRAPHLEARN_CORE_GRAPH_GRAPH_H_
#define GRAPHLEARN_CORE_GRAPH_GRAPH_H_

#include <vector>
#include <string>
#include "core/graph/storage/graph_storage.h"
#include "include/graph_request.h"
#include "include/index_option.h"
#include "include/status.h"

namespace graphlearn {

class Graph {
public:
  virtual ~Graph() = default;

  virtual Status Build(const IndexOption& option) = 0;

  virtual io::GraphStorage* GetLocalStorage() = 0;

#define DECLARE_METHOD(Name)                         \
  virtual Status Name(const Name##Request* request,  \
                      Name##Response* response) = 0; \
  virtual Status Name(int32_t remote_id,             \
                      const Name##Request* request,  \
                      Name##Response* response) = 0

  DECLARE_METHOD(UpdateEdges);
  DECLARE_METHOD(LookupEdges);

#undef DECLARE_METHOD
};

Graph* CreateLocalGraph(const std::string& type,
                        const std::string& view_type,
                        const std::string& use_attrs);
Graph* CreateRemoteGraph(const std::string& type,
                         const std::string& view_type,
                         const std::string& use_attrs);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_GRAPH_H_
