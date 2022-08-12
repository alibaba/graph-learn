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

  DagActorManager(const DagActorManager&) = delete;
  DagActorManager(DagActorManager&&) = delete;

  ~DagActorManager() {
    // clean up memory
    for (auto& param : params_) {
      delete param.second;
    }
  }

  DagActorManager& operator=(const DagActorManager&) = delete;
  DagActorManager& operator=(DagActorManager&&) = delete;

  void AddDag(const Dag* dag, uint32_t concurrency = 8) {
    // create actor ids and actor params
    auto dag_id = dag->Id();
    dag_actor_infos_[dag_id] = DagActorInfo{};
    auto& info = dag_actor_infos_[dag_id];

    // since we compute each op actor id use dag node id and dag id,
    // and dag node id is monotonically increasing from 1
    // for dag actor, we use dag nodes (size + 1, size + 1 + concurrency)
    // as the actor ids which is different from op actors
    for (int32_t i = 0; i < concurrency; ++i) {
      info.dag_actor_ids.push_back(
          MakeActorGUID(dag_id, dag->Nodes().size() + 1 + i));
    }

    for (auto& node : dag->Nodes()) {
      ActorIdType op_actor_id = MakeActorGUID(dag_id, node->Id());
      info.op_actor_ids[node->Id()] = op_actor_id;
      if (params_.count(op_actor_id) > 0) {
        delete params_[op_actor_id];
      }
      params_[op_actor_id] = new OpActorParams(node, op_actor_id);
    }

    for (auto dag_actor_id : info.dag_actor_ids) {
      if (params_.count(dag_actor_id) > 0) {
        delete params_[dag_actor_id];
      }
      params_[dag_actor_id] = new DagActorParams(dag, &info.op_actor_ids);
    }
  }

  void Clear() {
    dag_actor_infos_.clear();
    params_.clear();
  }

  const std::vector<ActorIdType>* GetDagActorIds(int32_t dag_id) const {
    return &dag_actor_infos_.at(dag_id).dag_actor_ids;
  }

  const ActorParams* GetActorParams(ActorIdType actor_id) const {
    return params_.at(actor_id);
  }

private:
  DagActorManager() = default;

private:
  struct DagActorInfo {
    std::vector<ActorIdType> dag_actor_ids;
    NodeIdToActorId          op_actor_ids;
    DagActorInfo() = default;
  };

  std::unordered_map<int32_t, DagActorInfo>     dag_actor_infos_;
  std::unordered_map<ActorIdType, ActorParams*> params_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_DAG_DAG_ACTOR_MANAGER_H_
