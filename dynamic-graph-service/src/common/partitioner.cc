/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "common/partitioner.h"

#include "common/log.h"

namespace dgs {

Partitioner PartitionerFactory::Create(const std::string& strategy,
                                       uint32_t num_partitions) {
  if (strategy == "hash") {
    return Partitioner{[n = num_partitions] (VertexId vid) {
      return (vid % n + n) % n;
    }, num_partitions};
  } else {
    std::string err_msg = "Unsupported partitioning strategy " + strategy;
    LOG(ERROR) << err_msg;
    throw std::runtime_error(err_msg);
  }
}

}  // namespace dgs
