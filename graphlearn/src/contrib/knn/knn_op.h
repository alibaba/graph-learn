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

#ifndef GRAPHLEARN_CONTRIB_KNN_KNN_OP_H_
#define GRAPHLEARN_CONTRIB_KNN_KNN_OP_H_

#include <string>
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/op_registry.h"
#include "graphlearn/include/status.h"

namespace graphlearn {
namespace op {

class KnnOperator : public RemoteOperator {
public:
  virtual ~KnnOperator() {}

  Status Process(const OpRequest* req,
                 OpResponse* res) override;

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_KNN_OP_H_
