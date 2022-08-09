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

#ifndef GRAPHLEARN_ACTOR_PARAMS_H_
#define GRAPHLEARN_ACTOR_PARAMS_H_

#include <cstdint>
#include <unordered_map>
#include <utility>

#include "hiactor/net/serializable_queue.hh"

#include "core/dag/dag.h"
#include "core/dag/dag_node.h"
#include "core/dag/tape.h"
#include "platform/env.h"

namespace graphlearn {
namespace act {

using ActorIdType   = uint32_t;
using DagNodeIdType = uint32_t;
using ShardIdType   = uint32_t;
using BatchIdType   = uint64_t;

using NodeIdToActorId = std::unordered_map<DagNodeIdType, ActorIdType>;

struct TapeActorConfig {
  Env* env;
  TapeStorePtr store;

  TapeActorConfig() : env(nullptr), store(nullptr) {}

  TapeActorConfig(TapeStorePtr st, Env* e) : env(e), store(std::move(st)) {}

  void dump_to(hiactor::serializable_queue& qu) {}
  static TapeActorConfig load_from(hiactor::serializable_queue &qu) {
    return {};
  }
};

struct ActorParams {
  const Dag* dag = nullptr;
  ActorParams() = default;
  explicit ActorParams(const Dag* dag) : dag(dag) {}
  virtual ~ActorParams() = default;
};

struct DagActorParams : public ActorParams {
  const NodeIdToActorId* node_id_to_actor_id;

  DagActorParams(const Dag* dag, const NodeIdToActorId* ids)
      : ActorParams(dag), node_id_to_actor_id(ids) {
  }
};

struct OpActorParams : public ActorParams {
  const DagNode* node;
  const ActorIdType self_actor_id;

  OpActorParams(const DagNode* node, ActorIdType actor_id)
      : ActorParams(), node(node), self_actor_id(actor_id) {
  }
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_PARAMS_H_
