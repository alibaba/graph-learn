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
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/graph_request.h"

namespace graphlearn {
namespace op {

class EdgeLookuper : public RemoteOperator {
public:
  virtual ~EdgeLookuper() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const LookupEdgesRequest* request =
      static_cast<const LookupEdgesRequest*>(req);
    LookupEdgesResponse* response =
      static_cast<LookupEdgesResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->EdgeType());
    return graph->LookupEdges(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const LookupEdgesRequest* request =
      static_cast<const LookupEdgesRequest*>(req);
    LookupEdgesResponse* response =
      static_cast<LookupEdgesResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->EdgeType());
    return graph->LookupEdges(remote_id, request, response);
  }
};

REGISTER_OPERATOR("LookupEdges", EdgeLookuper);

}  // namespace op
}  // namespace graphlearn
