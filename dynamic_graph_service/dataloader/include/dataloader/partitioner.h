/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DATALOADER_PARTITIONER_H_
#define DATALOADER_PARTITIONER_H_

#include <functional>
#include <vector>

#include "dataloader/typedefs.h"

namespace dgs {
namespace dataloader {

/// Output partition info manager
class Partitioner {
  using DataPartitionFunc = std::function<PartitionId(VertexId)>;
  using KafkaPartitionFunc = std::function<PartitionId(PartitionId)>;
public:
  static Partitioner& Get() {
    static Partitioner partitioner;
    return partitioner;
  }

  void Set(uint32_t data_partition_num, std::vector<PartitionId>&& kafka_router);

  PartitionId GetDataPartitionId(VertexId vid) const {
    return data_func_(vid);
  }

  PartitionId GetKafkaPartitionId(PartitionId data_partition) const {
    return kafka_func_(data_partition);
  }

private:
  Partitioner() = default;

private:
  DataPartitionFunc data_func_ = [] (VertexId vid) { return 0; };
  KafkaPartitionFunc kafka_func_ = [] (PartitionId data_partition) { return 0; };
};

inline
void Partitioner::Set(uint32_t data_partition_num, std::vector<PartitionId>&& kafka_router) {
  if (kafka_router.size() != data_partition_num) {
    throw std::runtime_error("Error init info with dataloader publish kafka partition map: size mismatch!");
  }
  data_func_ = [n = data_partition_num] (VertexId vid) {
    return (vid % n + n) % n;
  };
  kafka_func_ = [router = std::move(kafka_router)] (PartitionId data_partition) {
    return router[data_partition];
  };
}

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_PARTITIONER_H_
