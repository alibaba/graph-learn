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

#include "core/operator/graph/edge_generator.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace op {


class EdgeGetter : public RemoteOperator {
public:
  virtual ~EdgeGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetEdgesRequest* request =
      static_cast<const GetEdgesRequest*>(req);
    GetEdgesResponse* response =
      static_cast<GetEdgesResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->EdgeType());
    ::graphlearn::io::GraphStorage* storage = graph->GetLocalStorage();

    std::unique_ptr<EdgeGenerator> generator;
    if (request->Strategy() == "by_order") {
      generator.reset(new OrderedEdgeGenerator(storage));
    } else if (request->Strategy() == "random") {
      generator.reset(new RandomEdgeGenerator(storage));
    } else {
      generator.reset(new ShuffledEdgeGenerator(storage));
    }

    ::graphlearn::io::IdType src_id, dst_id, edge_id;
    int32_t expect_size = request->BatchSize();
    response->Init(expect_size);

    if (request->Epoch() < generator->Epoch()) {
      return error::OutOfRange("No more edges exist.");
    }

    for (int32_t i = 0; i < expect_size; ++i) {
      if (generator->Next(&src_id, &dst_id, &edge_id)) {
        response->Append(src_id, dst_id, edge_id);
      } else {
        break;
      }
    }

    if (response->Size() > 0) {
      return Status::OK();
    } else {
      // Begin next epoch.
      generator->Reset();
      generator->IncEpoch();
      return error::OutOfRange("No more edges exist.");
    }
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }
};

REGISTER_OPERATOR("GetEdges", EdgeGetter);

}  // namespace op
}  // namespace graphlearn
