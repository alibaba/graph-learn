/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/graph/graph_store.h"
#include "core/graph/storage/node_storage.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/graph_request.h"
#include "include/client.h"

namespace graphlearn {
namespace op {

class DegreeGetter : public RemoteOperator {
public:
  virtual ~DegreeGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetDegreeRequest* request =
      static_cast<const GetDegreeRequest*>(req);
    GetDegreeResponse* response =
      static_cast<GetDegreeResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->EdgeType());

    if (!graph) {
      LOG(ERROR) << "Edge type " << request->EdgeType() << " not existed.";
      return error::NotFound("Edge type not found.");
    }

    response->InitDegrees(request->BatchSize());

    if (request->GetNodeFrom() == NodeFrom::kEdgeSrc) {
      GetOutDegrees(graph, request, response);
    } else {
      return error::Unimplemented(
        "Get in_degrees is not implemented yet.");
    }
    return Status::OK();
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const GetDegreeRequest* request =
      static_cast<const GetDegreeRequest*>(req);
    GetDegreeResponse* response =
      static_cast<GetDegreeResponse*>(res);
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->GetDegree(request, response);
  }

private:
  Status GetOutDegrees(Graph* graph,
                       const GetDegreeRequest* request,
                       GetDegreeResponse* response) {
    ::graphlearn::io::GraphStorage* storage = graph->GetLocalStorage();
    const int64_t* ids = request->GetNodeIds();
    int32_t batch_size = request->BatchSize();
    for (int32_t i = 0; i < batch_size; ++i) {
      response->AppendDegree(storage->GetOutDegree(ids[i]));
    }
    return Status::OK();
  }
};

REGISTER_OPERATOR("GetDegree", DegreeGetter);

}  // namespace op
}  // namespace graphlearn
