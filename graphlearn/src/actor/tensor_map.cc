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

#include "actor/tensor_map.h"

#include "actor/tensor_serializer.h"

namespace graphlearn {
namespace act {

void TensorMap::dump_to(hiactor::serializable_queue& qu) {
  for (auto& it : tensors_) {
    // write key
    // FIXME(@goldenleaves): zero-copy.
    auto length = it.first.size();
    auto buf = seastar::temporary_buffer<char>(length);
    memcpy(buf.get_write(), it.first.data(), length);
    qu.push(std::move(buf));

    // write value
    TensorSerializer value_serializer(it.second);
    value_serializer.dump_to(qu);
  }
}

TensorMap TensorMap::load_from(hiactor::serializable_queue& qu) {
  TensorMap tm;
  while (!qu.empty()) {
    // get key
    auto buf = qu.pop();
    // FIXME(@goldenleaves): zero-copy.
    char *key_ptr = buf.get_write();
    auto key = std::string(buf.get_write(), buf.size());
    // get value
    auto value = TensorSerializer::load_from(qu);
    tm.tensors_[key] = std::move(value);
  }
  return tm;
}

ShardableTensorMap::ShardableTensorMap(bool shardable)
  : shardable_(shardable) {
}

ShardableTensorMap::ShardableTensorMap(Tensor::Map &&tensor,
                                       bool shardable)
  : tensors_(std::move(tensor)),
    shardable_(shardable) {
}

void ShardableTensorMap::SetPartitionKey(const std::string& pkey) {
  partition_key_ = pkey;
}

bool ShardableTensorMap::HasPartitionKey() const {
  return !partition_key_.empty();
}

ShardsPtr<ShardableTensorMap> ShardableTensorMap::Partition() const {
  // FIXME(@goldenleaves): Be careful, `n` should be initialized at runtime.
  static auto n = static_cast<int32_t>(hiactor::global_shard_count());
  static PartitionerCreator<ShardableTensorMap> creator(n);
  auto* partitioner = creator(GLOBAL_FLAG(PartitionMode));
  return partitioner->Partition(this);
}

void ShardableTensorMap::DisableShard() {
  shardable_ = false;
}

ShardableTensorMap* ShardableTensorMap::Clone() const {
  return new ShardableTensorMap;
}

JoinableTensorMap::JoinableTensorMap()
  : is_sparse_(false), batch_size_(-1) {
}

JoinableTensorMap::JoinableTensorMap(Tensor::Map &&tensor)
  : tensors_(std::move(tensor)),
    is_sparse_(false),
    batch_size_(-1) {
}

void JoinableTensorMap::SetSparseFlag() {
  is_sparse_ = true;
}

void JoinableTensorMap::Swap(JoinableTensorMap& right) {
  std::swap(batch_size_, right.batch_size_);
  std::swap(is_sparse_, right.is_sparse_);
  tensors_.swap(right.tensors_);
  params_.swap(right.params_);
}

void JoinableTensorMap::Stitch(ShardsPtr<JoinableTensorMap> shards) {
  auto stitcher = ::graphlearn::GetStitcher(this);
  stitcher->Stitch(std::move(shards), this);
}

}  // namespace act
}  // namespace graphlearn
