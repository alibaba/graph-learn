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

#include "actor/graph/sharded_graph_store.h"

#include "actor/graph/loader_status.h"
#include "actor/graph/loading_task.h"
#include "common/base/errors.h"
#include "core/io/slice_reader.h"
#include "include/config.h"
#include "platform/env.h"
#include "seastar/core/alien.hh"

namespace graphlearn {
namespace act {

ShardedGraphStore::ShardedGraphStore()
  : local_shards_(GLOBAL_FLAG(ActorLocalShardCount)),
    env_(nullptr), alien_tp_(nullptr) {}

ShardedGraphStore::~ShardedGraphStore() {
  for (auto store : store_group_) {
    delete store;
  }
}

void ShardedGraphStore::Init(Env* env) {
  env_ = env;
  store_group_.reserve(local_shards_);
  for (int32_t i = 0; i < local_shards_; ++i) {
    store_group_.push_back(new GraphStore(env_));
  }
}

void ShardedGraphStore::Finalize() {
  env_ = nullptr;
  for (auto store : store_group_) {
    delete store;
  }
  store_group_.clear();
}

Status ShardedGraphStore::Load(const std::vector<io::EdgeSource>& edges,
                               const std::vector<io::NodeSource>& nodes) {
  for (auto store : store_group_) {
    store->Init(edges, nodes);
  }
  InitAlienThread(edges, nodes);
  return Status::OK();
}

Status ShardedGraphStore::Build() {
  alien_tp_->initialize_and_run();
  DataLoaderStatus::Get()->WaitUntilFinished();
  alien_tp_.reset();

  return Status::OK();
}

void ShardedGraphStore::InitAlienThread(
    const std::vector<io::EdgeSource>& edges,
    const std::vector<io::NodeSource>& nodes) {
  int32_t batch_size = GLOBAL_FLAG(DataInitBatchSize);
  int32_t num_readers = std::max(GLOBAL_FLAG(InterThreadNum), 1);
  auto* tp = new hiactor::alien_thread_pool(num_readers);

  std::vector<io::NodeLoader*> node_loaders;
  std::vector<io::EdgeLoader*> edge_loaders;
  node_loaders.reserve(num_readers);
  edge_loaders.reserve(num_readers);

  for (int32_t i = 0; i < num_readers; ++i) {
    auto* node_reader = new io::NodeLoader(nodes, env_, i, num_readers);
    tp->add_task(new NodeLoadingTask(
        node_reader, i % local_shards_, batch_size));
    node_loaders.push_back(node_reader);
  }

  for (int32_t i = 0; i < num_readers; ++i) {
    auto* edge_reader = new io::EdgeLoader(edges, env_, i, num_readers);
    tp->add_task(new EdgeLoadingTask(
        edge_reader, i % local_shards_, batch_size));
    edge_loaders.push_back(edge_reader);
  }

  alien_tp_ = AlienTPUniquePtr{tp, ShardedGraphStore::ATPDeleter{
    std::move(node_loaders), std::move(edge_loaders)}};
}

void ShardedGraphStore::ATPDeleter::operator()(
    hiactor::alien_thread_pool* tp) {
  delete tp;
  for (auto &&nl : node_loaders_) {
    delete nl;
  }
  for (auto &&el : edge_loaders_) {
    delete el;
  }
}

}  // namespace act
}  // namespace graphlearn
