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

#ifndef GRAPHLEARN_CORE_OPERATOR_AGGREGATOR_AGGREGATOR_H_
#define GRAPHLEARN_CORE_OPERATOR_AGGREGATOR_AGGREGATOR_H_

#include <memory>
#include <string>
#include <vector>
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/aggregating_request.h"
#include "graphlearn/include/status.h"
#include "graphlearn/include/client.h"

namespace graphlearn {
namespace op {

class Aggregator : public RemoteOperator {
public:
  virtual ~Aggregator() {}

  Status Process(const OpRequest* req, OpResponse* res) override {
    const AggregatingRequest* request =
      static_cast<const AggregatingRequest*>(req);
    AggregatingResponse* response =
      static_cast<AggregatingResponse*>(res);
    return this->Aggregate(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const AggregatingRequest* request =
      static_cast<const AggregatingRequest*>(req);
    AggregatingResponse* response =
      static_cast<AggregatingResponse*>(res);
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->Aggregating(request, response);
  }

public:
  virtual Status Aggregate(const AggregatingRequest* req,
                           AggregatingResponse* res);

  virtual void InitFunc(float* value,
                        int32_t dim);
  virtual void AggFunc(float* left,
                       const float* right,
                       int32_t size,
                       const int32_t* segments = nullptr,
                       int32_t num_segments = 0);
  virtual void FinalFunc(float* values,
                         int32_t size,
                         const int32_t* segments,
                         int32_t num_segments);
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_AGGREGATOR_AGGREGATOR_H_
