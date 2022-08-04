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

#ifndef GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_MANAGER_H_
#define GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_MANAGER_H_

#include <unordered_map>

#include "actor/params.h"
#include "actor/utils.h"
#include "core/dag/dag.h"

namespace graphlearn {
namespace act {

class DagActorManager {
public:
  static DagActorManager& GetInstance() {
    static DagActorManager instance;
    return instance;
  }

  ~DagActorManager() {
    // clean up memory
    for (auto& param : params_) {
      delete param.second;
    }
  }

  void Init(const Dag* dag, uint32_t concurrency = 8) {
    // create actor ids and actor params
    auto dag_id = dag->Id();
    // since we compute each op actor id use dag node id and dag id,
    // and dag node id is monotonically increasing from 1
    // for dag actor, we use dag nodes (size + 1, size + 1 + concurrency)
    // as the actor ids which is different from op actors
    for (int32_t i = 0; i < concurrency; ++i) {
      dag_actor_ids_.push_back(
          MakeActorGUID(dag_id, dag->Nodes().size() + 1 + i));
    }

    for (auto& node : dag->Nodes()) {
      ActorIdType actor_id = MakeActorGUID(dag_id, node->Id());
      op_actor_ids_[node->Id()] = actor_id;
      params_[actor_id] = new OpActorParams(node, actor_id);
    }
    for (auto dag_actor_id : dag_actor_ids_) {
      params_[dag_actor_id] = new DagActorParams(dag, &op_actor_ids_);
    }
  }

  const std::vector<ActorIdType>* GetDagActorIds() const {
    return &dag_actor_ids_;
  }

  const ActorParams* GetActorParams(ActorIdType actor_id) const {
    return params_.at(actor_id);
  }

  const NodeIdToActorId* GetOpActorIds() const {
    return &op_actor_ids_;
  }

private:
  DagActorManager() = default;

private:
  NodeIdToActorId          op_actor_ids_;
  std::vector<ActorIdType> dag_actor_ids_;
  std::unordered_map<ActorIdType, ActorParams*> params_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_MANAGER_H_
