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

#ifndef GRAPHLEARN_ACTOR_GRAPH_SHARDED_GRAPH_STORE_H_
#define GRAPHLEARN_ACTOR_GRAPH_SHARDED_GRAPH_STORE_H_

#include <memory>
#include <utility>
#include <vector>
#include "brane/core/alien_thread_pool.hh"
#include "core/graph/graph_store.h"
#include "core/io/edge_loader.h"
#include "core/io/node_loader.h"

namespace graphlearn {

class Env;

namespace actor {

class ShardedGraphStore {
public:
  static inline ShardedGraphStore& Get() {
    static ShardedGraphStore instance;
    return instance;
  }

  inline GraphStore* OnShard(int32_t shard_id) {
    return store_group_[shard_id];
  }

  void Init(Env* env);

  Status Load(const std::vector<io::EdgeSource>& edges,
              const std::vector<io::NodeSource>& nodes);

  Status Build();

private:
  ShardedGraphStore();
  ~ShardedGraphStore();

  void Finalize();
  void InitAlienThread(const std::vector<io::EdgeSource>& edges,
                       const std::vector<io::NodeSource>& nodes);

private:
  struct ATPDeleter {
    ATPDeleter() = default;
    ATPDeleter(std::vector<io::NodeLoader*> &&nls,
               std::vector<io::EdgeLoader*> &&els)
      : node_loaders_(std::move(nls)),
        edge_loaders_(std::move(els)) {}

    void operator()(brane::alien_thread_pool* tp);

  private:
    std::vector<io::NodeLoader*> node_loaders_;
    std::vector<io::EdgeLoader*> edge_loaders_;
  };

  using AlienTPUniquePtr = std::unique_ptr<
    brane::alien_thread_pool, ATPDeleter>;

private:
  int32_t                  local_shards_;
  Env*                     env_;
  std::vector<GraphStore*> store_group_;
  AlienTPUniquePtr         alien_tp_;

  friend class ActorService;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_SHARDED_GRAPH_STORE_H_
