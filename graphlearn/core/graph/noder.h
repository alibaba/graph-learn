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

#ifndef GRAPHLEARN_CORE_GRAPH_NODER_H_
#define GRAPHLEARN_CORE_GRAPH_NODER_H_

#include <vector>
#include <string>
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/include/graph_request.h"
#include "graphlearn/include/status.h"

namespace graphlearn {

class Noder {
public:
  virtual ~Noder() = default;

  virtual void Build() = 0;

  virtual io::NodeStorage* GetLocalStorage() = 0;

#define DECLARE_METHOD(Name)                         \
  virtual Status Name(const Name##Request* request,  \
                      Name##Response* response) = 0; \
  virtual Status Name(int32_t remote_id,             \
                      const Name##Request* request,  \
                      Name##Response* response) = 0

  DECLARE_METHOD(UpdateNodes);
  DECLARE_METHOD(LookupNodes);

#undef DECLARE_METHOD
};

Noder* CreateLocalNoder(const std::string& type,
                        const std::string& view_type,
                        const std::string &use_attrs);
Noder* CreateRemoteNoder(const std::string& type,
                         const std::string& view_type,
                         const std::string &use_attrs);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_NODER_H_
