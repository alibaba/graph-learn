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

#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/op_registry.h"
#include "graphlearn/include/graph_request.h"

namespace graphlearn {
namespace op {

class EdgeCountGetter : public RemoteOperator {
public:
  virtual ~EdgeCountGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetCountRequest* request =
      static_cast<const GetCountRequest*>(req);
    GetCountResponse* response =
      static_cast<GetCountResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->Type());
    ::graphlearn::io::GraphStorage* storage = graph->GetLocalStorage();

    response->Init();
    response->Set(storage->GetEdgeCount());
    return Status::OK();
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }
};

REGISTER_OPERATOR("GetEdgeCount", EdgeCountGetter);

}  // namespace op
}  // namespace graphlearn
