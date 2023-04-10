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

#include "common/base/errors.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "core/operator/graph/node_generator.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace op {

class NodeGetter : public RemoteOperator {
public:
  virtual ~NodeGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetNodesRequest* request =
      static_cast<const GetNodesRequest*>(req);
    GetNodesResponse* response =
      static_cast<GetNodesResponse*>(res);

    StorageWrapper* storage =
      new StorageWrapper(request->GetNodeFrom(), request->Type(), graph_store_);
    std::unique_ptr<Generator> generator = GetGenerator(
      storage, request->Strategy());
    return GetNode(generator, request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }


private:
  std::unique_ptr<Generator> GetGenerator(
      StorageWrapper* storage, const std::string& strategy) {
    std::unique_ptr<Generator> generator;
    if (strategy == "by_order") {
      generator.reset(new OrderedGenerator(storage));
    } else if (strategy == "random") {
      generator.reset(new RandomGenerator(storage));
    } else {
      generator.reset(new ShuffledGenerator(storage));
    }
    return generator;
  }

  Status GetNode(const std::unique_ptr<Generator>&
      generator, const GetNodesRequest* request,
      GetNodesResponse* response) {
    ::graphlearn::io::IdType id = 0;
    int32_t expect_size = request->BatchSize();
    response->Init(expect_size);

    if (request->Epoch() < generator->Epoch()) {
      return error::OutOfRange("No more nodes exist.");
    }

    for (int32_t i = 0; i < expect_size; ++i) {
      if (generator->Next(&id)) {
        response->Append(id);
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
      return error::OutOfRange("No more nodes exist.");
    }
    return Status::OK();
  }
};

REGISTER_OPERATOR("GetNodes", NodeGetter);

}  // namespace op
}  // namespace graphlearn
