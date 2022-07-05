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

#ifndef GRAPHLEARN_CORE_OPERATOR_UTILS_GET_NODE_ATTRIBUTES_WRAPPER_H_
#define GRAPHLEARN_CORE_OPERATOR_UTILS_GET_NODE_ATTRIBUTES_WRAPPER_H_

#include <string>

#include "include/client.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace op {

class GetNodeAttributesWrapper {
public:
  GetNodeAttributesWrapper(const std::string& node_type,
                           const int64_t* id,
                           int32_t batch_size);
  ~GetNodeAttributesWrapper();

  const Status& GetStatus();
  const int64_t* NextIntAttrs();
  const float* NextFloatAttrs();
  const std::string* const* NextStrAttrs();

private:
  Status Lookup(const std::string& node_type,
                const int64_t* id,
                int32_t batch_size);

private:
  Status               status_;
  LookupNodesRequest*  req_;
  LookupNodesResponse* res_;
  int32_t              i_cur_;
  int32_t              i_num_;
  int32_t              f_cur_;
  int32_t              f_num_;
  int32_t              s_cur_;
  int32_t              s_num_;
};


}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_UTILS_GET_NODE_ATTRIBUTES_WRAPPER_H_
