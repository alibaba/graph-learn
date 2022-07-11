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

#ifndef DGS_COMMON_PARTITIONER_H_
#define DGS_COMMON_PARTITIONER_H_

#include <functional>
#include <string>

#include "common/typedefs.h"

namespace dgs {

class Partitioner {
  using Func = std::function<PartitionId(VertexId)>;
public:
  Partitioner() = default;
  Partitioner(Func&& func, uint32_t num_partitions) :
    func_(std::move(func)), num_partitions_(num_partitions) {}

  Partitioner(Partitioner&& other) = default;
  Partitioner& operator=(Partitioner&& other) = default;

  PartitionId GetPartitionId(VertexId vid) const {
    return func_(vid);
  }

  uint32_t GetPartitionsNum() {
    return num_partitions_;
  }

private:
  Func      func_;
  uint32_t  num_partitions_;
};

struct PartitionerFactory {
public:
  static Partitioner Create(const std::string& strategy,
                            uint32_t num_partitions);
};

}  // namespace dgs

#endif  // DGS_COMMON_PARTITIONER_H_
