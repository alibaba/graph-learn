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

#ifndef GRAPHLEARN_SERVICE_DIST_LOAD_BALANCER_H_
#define GRAPHLEARN_SERVICE_DIST_LOAD_BALANCER_H_

#include <cstdint>
#include <vector>
#include "graphlearn/include/status.h"

namespace graphlearn {

/// Here we abstract the servers as resources. The server number is
/// resource number. LoadBalancer is designed to find one or more
/// available servers for the current client.
class LoadBalancer {
public:
  explicit LoadBalancer(int32_t resource_num)
      : resource_num_(resource_num) {}
  virtual ~LoadBalancer() = default;

  /// Split total resources into part_num, and each part has at least
  /// replica resources.
  virtual Status Calc(int32_t part_num, int32_t replica) = 0;

  /// Get resource ids for the given part.
  virtual Status GetPart(int32_t part_id,
                         std::vector<int32_t>* resources) = 0;
protected:
  int32_t resource_num_;
};

LoadBalancer* NewRoundRobinBalancer(int32_t resource_num);

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_LOAD_BALANCER_H_
