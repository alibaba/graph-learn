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

#ifndef GRAPHLEARN_ACTOR_TENSOR_MAP_H_
#define GRAPHLEARN_ACTOR_TENSOR_MAP_H_

#include <string>
#include <utility>
#include <unordered_map>

#include "hiactor/core/shard-config.hh"
#include "hiactor/net/serializable_queue.hh"

#include "include/tensor.h"
#include "core/dag/tape.h"
#include "core/partition/partitioner.h"

namespace graphlearn {
namespace act {

struct TapeHolder {
  Tape* tape;

  explicit TapeHolder(Tape* t) : tape(t) {}
  ~TapeHolder() = default;

  // TapeHolder doesn't need (de)serialization.
  void dump_to(hiactor::serializable_queue& qu) {}
  static TapeHolder load_from(hiactor::serializable_queue& qu) {
    return TapeHolder{nullptr};
  }
};

class TensorMap {
public:
  TensorMap() = default;

  explicit TensorMap(Tensor::Map&& tensor)
    : tensors_(std::move(tensor)) {
  }

  TensorMap(TensorMap&& other) noexcept
    : tensors_(std::move(other.tensors_)) {
  }

  TensorMap& operator=(TensorMap&& other) noexcept {
    if (this != &other) {
      tensors_ = std::move(other.tensors_);
    }
    return *this;
  }

  void dump_to(hiactor::serializable_queue& qu);
  static TensorMap load_from(hiactor::serializable_queue& qu);

  size_t Size() const { return tensors_.size(); }

public:
  Tensor::Map tensors_;
};

class ShardableTensorMap {
public:
  explicit ShardableTensorMap(bool shardable = true);
  explicit ShardableTensorMap(Tensor::Map&& tensor, bool shardable = true);

  size_t Size() const { return tensors_.size(); }

  void SetPartitionKey(const std::string& pkey);
  const std::string& PartitionKey() const { return partition_key_; }
  bool HasPartitionKey() const;
  static uint32_t PartitionId() { return hiactor::global_shard_id(); }
  ShardsPtr<ShardableTensorMap> Partition() const;

  void DisableShard();

  ShardableTensorMap* Clone() const;

private:
  bool shardable_;
  std::string partition_key_;

// fixme: actor
public:
  Tensor::Map tensors_;
};

class JoinableTensorMap {
public:
  JoinableTensorMap();
  explicit JoinableTensorMap(Tensor::Map&& tensor);

  size_t Size() const { return tensors_.size(); }

  void SetSparseFlag();
  bool IsSparse() const { return is_sparse_; }

  void Swap(JoinableTensorMap& right);
  // fixme: actor, pass by value?
  void Stitch(ShardsPtr<JoinableTensorMap> shards);

private:
  bool is_sparse_;

public:
  int32_t     batch_size_;
  Tensor::Map params_;
  Tensor::Map tensors_;
};

template<class T>
BasePartitioner<T>* GetPartitioner(const T* t) {
  // FIXME(@goldenleaves): Be careful, `n` should be initialized at runtime.
  static auto n = static_cast<int32_t>(hiactor::global_shard_count());
  static PartitionerCreator<T> creator(n);
  return creator(GLOBAL_FLAG(PartitionMode));
}

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_TENSOR_MAP_H_
