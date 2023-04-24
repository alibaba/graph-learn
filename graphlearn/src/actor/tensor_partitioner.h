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

#ifndef GRAPHLEARN_ACTOR_TENSOR_PARTITIONER_H_
#define GRAPHLEARN_ACTOR_TENSOR_PARTITIONER_H_

#include <string>
#include <utility>
#include <unordered_map>

#include "hiactor/core/shard-config.hh"
#include "hiactor/net/serializable_queue.hh"

#include "include/tensor.h"
#include "core/dag/tensor_map.h"
#include "core/partition/partitioner.h"

namespace graphlearn {
namespace act {
class ShardableTensorMap {
public:
  explicit ShardableTensorMap(const std::string& shard_key,
                              bool shardable = true);
  ShardableTensorMap(const std::string& shard_key,
                     TensorMap&& tensor,
                     bool shardable = true);

  int32_t Size() { return tensors_.size(); }

  bool IsShardable() const { return shardable_; }
  const std::string& ShardKey() const { return shard_key_; }
  int32_t ShardId() const { return hiactor::global_shard_id(); }
  ShardsPtr<ShardableTensorMap> Partition() const;

  void DisableShard();

  ShardableTensorMap* Clone() const;

private:
  const std::string shard_key_;
  bool shardable_;

public:
  Tensor::Map tensors_;
  SparseTensor::Map sparse_tensors_;
};

class JoinableTensorMap {
public:
  JoinableTensorMap();
  explicit JoinableTensorMap(TensorMap&& tensor);

  int32_t Size() { return tensors_.size();}

  void Swap(JoinableTensorMap& right);
  void Stitch(ShardsPtr<JoinableTensorMap> shards);

public:
  int32_t     batch_size_;
  Tensor::Map params_;
  // TODO(wenting.swt): replace me with TensorMap
  Tensor::Map tensors_;
  SparseTensor::Map sparse_tensors_;
};

template<class T>
BasePartitioner<T>* GetPartitioner(const T* t) {
  static int32_t n = hiactor::global_shard_count();
  static PartitionerCreator<T> creator(n);
  return creator(GLOBAL_FLAG(PartitionMode));
}

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_TENSOR_PARTITIONER_H_
