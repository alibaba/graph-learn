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

#ifndef GRAPHLEARN_ACTOR_DAG_DAG_PROXY_H_
#define GRAPHLEARN_ACTOR_DAG_DAG_PROXY_H_

#include <queue>
#include <vector>

#include "actor/params.h"
#include "core/dag/dag.h"
#include "core/dag/dag_edge.h"
#include "core/dag/dag_node.h"

namespace graphlearn {
namespace act {

class BaseOperatorActor_ref;

class EdgeProxy {
public:
  EdgeProxy() = default;
  explicit EdgeProxy(DagEdgePtr e);
  EdgeProxy(EdgeProxy&& other) noexcept;

  DagNodeIdType UpstreamGUID() const;
  std::pair<std::string, std::string> Joint() const;

private:
  DagEdgePtr edge_;
};

class NodeProxy {
public:
  NodeProxy() : node_(nullptr), actor_id_(0), shard_key_("") {}
  explicit NodeProxy(const DagNode* n, ActorIdType actor_id);
  NodeProxy(NodeProxy&& other) noexcept;
  ~NodeProxy();

  DagNodeIdType GUID() const {
    return node_->Id();
  }

  const std::string& OpName() const {
    return node_->OpName();
  }

  const Tensor::Map& Params() const {
    return node_->Params();
  }

  const std::string& ShardKey() const {
    return shard_key_;
  }

  const std::vector<EdgeProxy>& Upstreams() const {
    return upstreams_;
  }

  bool IsSource() const {
    return node_->Id() == 1;
  }

  BaseOperatorActor_ref* OnShard(uint32_t shard_id) const {
    return actor_refs_[shard_id];
  }

private:
  void InitActorRef(uint32_t total_shards);

private:
  const DagNode* node_;
  std::string shard_key_;
  std::vector<EdgeProxy> upstreams_;
  ActorIdType actor_id_;
  /// Actor ref for each shard
  std::vector<BaseOperatorActor_ref*> actor_refs_;
};

class DagProxy {
public:
  DagProxy() = default;
  explicit DagProxy(const DagActorParams* params);

  bool HasNext() const;
  /// Return a vector of vertex ids that are ready to run.
  std::vector<DagNodeIdType> Next();

  /// Return a vertex proxy with given id.
  NodeProxy& Node(DagNodeIdType id) {
    return nodes_[id];
  }

  /// Reset the internal info to execute the DAG once again.
  void Reset();

private:
  void AddNodeRecursive(const DagNode* node);
  /// Add a vertex.
  void AddNode(DagNodeIdType id, NodeProxy&& node);
  /// Add edge s -> t. We assume that s and t have already been added.
  void AddEdge(DagNodeIdType s, DagNodeIdType t);

private:
  int32_t dag_id_ = 0;
  const std::unordered_map<DagNodeIdType, ActorIdType>* node_id_to_actor_id_ = nullptr;
  // Use adjacency matrix to store the DAG.
  // guid -> list of neighbors
  std::unordered_map<DagNodeIdType, std::vector<DagNodeIdType>> adj_;
  std::unordered_map<DagNodeIdType, NodeProxy> nodes_;
  // guid -> number of in edges
  std::unordered_map<DagNodeIdType, int32_t> in_degree_;
  // Used for reset the dag
  std::unordered_map<DagNodeIdType, int32_t> in_degree_inited_;
  // Store the node ids that are ready to run
  std::queue<DagNodeIdType> ready_;
  // The number of nodes that are processed
  int processed_ = 0;

  friend class DagActor;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_DAG_DAG_PROXY_H_
