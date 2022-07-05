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

class StatsGetter : public RemoteOperator {
public:
  virtual ~StatsGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetStatsRequest* request =
      static_cast<const GetStatsRequest*>(req);
    GetStatsResponse* response =
      static_cast<GetStatsResponse*>(res);
    if (graph_store_->GetStatistics().GetCounts().empty()) {
      graph_store_->BuildStatistics();
    }
    response->SetCounts(graph_store_->GetStatistics().GetCounts());
    return Status::OK();
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }
};

REGISTER_OPERATOR("GetStats", StatsGetter);

}  // namespace op
}  // namespace graphlearn
