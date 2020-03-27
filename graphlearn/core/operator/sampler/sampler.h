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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_SAMPLER_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_SAMPLER_H_

#include <memory>
#include <string>
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/sampling_request.h"
#include "graphlearn/include/status.h"
#include "graphlearn/include/client.h"

namespace graphlearn {
namespace op {

/// For types of samplers, we customize a class from operator.
/// Each new extended sampler just needs to inherit from this class.
class Sampler : public RemoteOperator {
public:
  virtual ~Sampler() {}

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const SamplingRequest* request =
      static_cast<const SamplingRequest*>(req);
    SamplingResponse* response =
      static_cast<SamplingResponse*>(res);

    return this->Sample(request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const SamplingRequest* request =
      static_cast<const SamplingRequest*>(req);
    SamplingResponse* response =
      static_cast<SamplingResponse*>(res);
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->Sampling(request, response);
  }

protected:
  virtual Status Sample(const SamplingRequest* req,
                        SamplingResponse* res) = 0;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_SAMPLER_H_
