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

#include "core/graph/graph_store.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace op {

class NodeLookuper : public RemoteOperator {
public:
  virtual ~NodeLookuper() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const LookupNodesRequest* request =
      static_cast<const LookupNodesRequest*>(req);
    LookupNodesResponse* response =
      static_cast<LookupNodesResponse*>(res);

    Noder* noder = graph_store_->GetNoder(request->NodeType());
    return noder->LookupNodes(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const LookupNodesRequest* request =
      static_cast<const LookupNodesRequest*>(req);
    LookupNodesResponse* response =
      static_cast<LookupNodesResponse*>(res);

    Noder* noder = graph_store_->GetNoder(request->NodeType());
    return noder->LookupNodes(remote_id, request, response);
  }
};

REGISTER_OPERATOR("LookupNodes", NodeLookuper);

}  // namespace op
}  // namespace graphlearn
