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
#include "core/io/element_value.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace op {

class NodeUpdater : public RemoteOperator {
public:
  virtual ~NodeUpdater() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const UpdateNodesRequest* request =
      static_cast<const UpdateNodesRequest*>(req);
    UpdateNodesResponse* response =
      static_cast<UpdateNodesResponse*>(res);

    const ::graphlearn::io::SideInfo* info = request->GetSideInfo();
    Noder* noder = graph_store_->GetNoder(info->type);
    return noder->UpdateNodes(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const UpdateNodesRequest* request =
      static_cast<const UpdateNodesRequest*>(req);
    UpdateNodesResponse* response =
      static_cast<UpdateNodesResponse*>(res);

    const ::graphlearn::io::SideInfo* info = request->GetSideInfo();
    Noder* noder = graph_store_->GetNoder(info->type);
    return noder->UpdateNodes(remote_id, request, response);
  }
};

REGISTER_OPERATOR("UpdateNodes", NodeUpdater);

}  // namespace op
}  // namespace graphlearn
