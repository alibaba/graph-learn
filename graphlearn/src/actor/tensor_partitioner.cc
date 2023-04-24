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

#include "actor/tensor_partitioner.h"

namespace graphlearn {
namespace act {

ShardableTensorMap::ShardableTensorMap(const std::string& shard_key,
                                       bool shardable)
  : shard_key_(shard_key),
    shardable_(true) {
}

ShardableTensorMap::ShardableTensorMap(const std::string& shard_key,
                                       TensorMap&& tensor,
                                       bool shardable)
  : shard_key_(shard_key),
    tensors_(std::move(tensor.tensors_)),
    sparse_tensors_(std::move(tensor.sparse_tensors_)),
    shardable_(shardable) {
}

ShardsPtr<ShardableTensorMap> ShardableTensorMap::Partition() const {
  static int32_t n = hiactor::global_shard_count();
  static PartitionerCreator<ShardableTensorMap> creator(n);
  auto* partitioner = creator(GLOBAL_FLAG(PartitionMode));
  return partitioner->Partition(this);
}

void ShardableTensorMap::DisableShard() {
  shardable_ = false;
}

ShardableTensorMap* ShardableTensorMap::Clone() const {
  ShardableTensorMap* req = new ShardableTensorMap(shard_key_);
  return req;
}

JoinableTensorMap::JoinableTensorMap()
  : batch_size_(-1) {
}

JoinableTensorMap::JoinableTensorMap(TensorMap&& tensor)
  : tensors_(std::move(tensor.tensors_)),
    sparse_tensors_(std::move(tensor.sparse_tensors_)),
    batch_size_(-1) {
}

void JoinableTensorMap::Swap(JoinableTensorMap& right) {
  std::swap(batch_size_, right.batch_size_);
  // TODO(wenting.swt): check me if is right
  tensors_.swap(right.tensors_);
  sparse_tensors_.swap(right.sparse_tensors_);
  params_.swap(right.params_);
}

void JoinableTensorMap::Stitch(ShardsPtr<JoinableTensorMap> shards) {
  auto stitcher = ::graphlearn::GetStitcher(this);
  stitcher->Stitch(shards, this);
}

}  // namespace act
}  // namespace graphlearn
