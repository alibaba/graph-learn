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
#include "graphlearn/core/operator/op_registry.h"
#include "graphlearn/include/graph_request.h"

namespace graphlearn {
namespace op {

class CountGetter : public RemoteOperator {
public:
  virtual ~CountGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetCountRequest* request =
      static_cast<const GetCountRequest*>(req);
    GetCountResponse* response =
      static_cast<GetCountResponse*>(res);
    const auto& local_count = graph_store_->GetLocalCount();
    response->Init(local_count.size());
    for (const auto& count : local_count) {
      response->Append(count);
    }
    return Status::OK();
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }
};

REGISTER_OPERATOR("GetCount", CountGetter);

}  // namespace op
}  // namespace graphlearn
