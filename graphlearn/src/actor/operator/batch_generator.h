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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_BATCH_GENERATOR_H_
#define GRAPHLEARN_ACTOR_OPERATOR_BATCH_GENERATOR_H_

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "actor/graph/sharded_graph_store.h"
#include "actor/tensor_map.h"
#include "actor/utils.h"
#include "core/graph/graph_store.h"
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace actor {

extern const char* DelegateFetchFlag;

struct BatchLocation {
  uint32_t shard_id;
  uint32_t offset;
  uint32_t length;
  BatchLocation(uint32_t sid, uint32_t offset, uint32_t length)
    : shard_id(sid), offset(offset), length(length) {}
  ~BatchLocation() = default;
};

class NodeBatchGenerator {
public:
  NodeBatchGenerator() = default;
  virtual ~NodeBatchGenerator() = default;
  virtual seastar::future<TensorMap> NextBatch() = 0;

protected:
  ShardDataInfoVecT GetSortedDataInfos(const std::string& type);
};

class TraverseNodeBatchGenerator : public NodeBatchGenerator {
private:
  struct Iterator;
  struct OrderedIterator;
  struct ShuffledIterator;

public:
  TraverseNodeBatchGenerator(const std::string& type, unsigned batch_size,
    const OpActorParams* params, const std::string& strategy);
  ~TraverseNodeBatchGenerator() override;
  seastar::future<TensorMap> NextBatch() override;

private:
  Iterator* iter_;
};

class RandomNodeBatchGenerator : public NodeBatchGenerator {
public:
  RandomNodeBatchGenerator(const std::string& type, unsigned batch_size);
  ~RandomNodeBatchGenerator() override {}
  seastar::future<TensorMap> NextBatch() override;

private:
  const io::IdList*  ids_;
  const unsigned     batch_size_;
  unsigned           data_size_;
  std::random_device rd_;
  std::mt19937       engine_;
  std::uniform_int_distribution<int32_t> dist_;
};

class EdgeBatchGenerator {
public:
  EdgeBatchGenerator() = default;
  virtual ~EdgeBatchGenerator() = default;
  virtual seastar::future<TensorMap> NextBatch() = 0;

protected:
  ShardDataInfoVecT GetSortedDataInfos(const std::string& type);
};

class TraverseEdgeBatchGenerator : public EdgeBatchGenerator {
private:
  struct Iterator;
  struct OrderedIterator;
  struct ShuffledIterator;

public:
  TraverseEdgeBatchGenerator(const std::string& type, unsigned batch_size,
    const OpActorParams* params, const std::string& strategy);
  ~TraverseEdgeBatchGenerator() override;
  seastar::future<TensorMap> NextBatch() override;

private:
  Iterator* iter_;
};

class RandomEdgeBatchGenerator : public EdgeBatchGenerator {
public:
  RandomEdgeBatchGenerator(const std::string& type, unsigned batch_size);
  ~RandomEdgeBatchGenerator() override {}
  seastar::future<TensorMap> NextBatch() override;

private:
  const unsigned                  batch_size_;
  ::graphlearn::io::GraphStorage* storage_;
  ::graphlearn::io::IdType        edge_count_;
  std::random_device              rd_;
  std::mt19937                    engine_;
  std::uniform_int_distribution<::graphlearn::io::IdType> dist_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_BATCH_GENERATOR_H_
