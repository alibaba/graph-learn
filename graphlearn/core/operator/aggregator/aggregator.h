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

#include <string>
#include <vector>
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/aggregation_request.h"
#include "graphlearn/include/status.h"

namespace graphlearn {
namespace op {

class Aggregator : public RemoteOperator {
public:
  virtual ~Aggregator() {}

  Status Process(const OpRequest* req, OpResponse* res) override {
    const AggregateNodesRequest* request =
      static_cast<const AggregateNodesRequest*>(req);
    AggregateNodesResponse* response =
      static_cast<AggregateNodesResponse*>(res);
    return this->Aggregate(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    // interface for remote call, now just use local instead.
    return Process(req, res);
  }

protected:
  virtual Status Aggregate(const AggregateNodesRequest* req,
                           AggregateNodesResponse* res);

  virtual void InitFunc(std::vector<float>* value,
                        int32_t size);
  virtual void AggFunc(std::vector<float>* left,
                       const std::vector<float>& right);
  virtual void FinalFunc(std::vector<float>* values,
                         int32_t total);
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_AGGREGATOR_AGGREGATOR_H_
