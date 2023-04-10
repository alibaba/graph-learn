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

#include "actor/dag/dag_proxy.h"

#include "actor/operator/op_ref_factory.h"
#include "actor/utils.h"

namespace graphlearn {
namespace act {

EdgeProxy::EdgeProxy(DagEdgePtr e) : edge_(std::move(e)) {
}

EdgeProxy::EdgeProxy(EdgeProxy&& other) noexcept
    : edge_(std::move(other.edge_)) {
}

DagNodeIdType EdgeProxy::UpstreamGUID() const {
  return edge_->Src()->Id();
}

std::pair<std::string, std::string> EdgeProxy::Joint() const {
  return { edge_->SrcOutput(), edge_->DstInput() };
}

NodeProxy::NodeProxy(const DagNode* n, ActorIdType actor_id)
    : node_(n), actor_id_(actor_id) {
  for (const auto& edge : node_->InEdges()) {
    upstreams_.emplace_back(edge);
  }
  InitActorRef(hiactor::global_shard_count());

  shard_key_ = RequestFactory::GetInstance()->NewRequest(node_->OpName())->ShardKey();
}

NodeProxy::NodeProxy(NodeProxy&& other) noexcept
  : node_(other.node_),
    shard_key_(other.shard_key_),
    actor_id_(other.actor_id_),
    upstreams_(std::move(other.upstreams_)),
    actor_refs_(std::move(other.actor_refs_)) {
}

NodeProxy::~NodeProxy() {
  for (auto ref : actor_refs_) {
    delete ref;
  }
}

void NodeProxy::InitActorRef(uint32_t total_shards) {
  actor_refs_.resize(total_shards, nullptr);
  std::string op_name = node_->OpName();
  for (int32_t shard_id = 0; shard_id < total_shards; ++shard_id) {
    auto builder = hiactor::scope_builder(shard_id);
    actor_refs_[shard_id] = OpRefFactory::Get().Create(
      op_name, actor_id_, &builder);
  }
}

DagProxy::DagProxy(const DagActorParams* params) {
  const Dag* dag = params->dag;
  node_id_to_actor_id_ = params->node_id_to_actor_id;
  dag_id_ = dag->Id();
  /// Add indegree for root node if the dag has only one executed node
  /// and a sink node.
  if (dag->Size() == 2) {
    in_degree_.insert(std::make_pair(1, 0));
  }
  AddNodeRecursive(dag->Root());
  in_degree_inited_ = in_degree_;
}

bool DagProxy::HasNext() const {
  return processed_ < adj_.size();
}

std::vector<DagNodeIdType> DagProxy::Next() {
  if (ready_.empty()) {
    // Push the nodes that are ready to run to the queue.
    // ready_.empty() is only in the first time.
    for (auto it : in_degree_) {
      if (it.second == 0) {
        ready_.push(it.first);
      }
    }
  }

  std::vector<DagNodeIdType> result;
  auto size = ready_.size();
  // Pop the nodes that are ready to run in this round.
  for (int i = 0; i < size; ++i) {
    DagNodeIdType ready_node = ready_.front();
    ready_.pop();
    result.push_back(ready_node);
    ++processed_;

    // Decrease the indegree of the node's neighbors.
    // if inDegree of a node i's neighbor is 0, add to the ready queue.
    // And then it will be popped in the next round.
    for (auto neighbor : adj_[ready_node]) {
      in_degree_[neighbor]--;
      if (in_degree_[neighbor] == 0) {
        ready_.push(neighbor);
      }
    }
  }
  return result;
}

void DagProxy::Reset() {
  in_degree_ = in_degree_inited_;
  processed_ = 0;
}

void DagProxy::AddNodeRecursive(const DagNode* node) {
  if (node == nullptr || node->IsSink()) {
    return;
  }

  AddNode(node->Id(), NodeProxy(node, node_id_to_actor_id_->at(node->Id())));

  auto down_streams = node->OutEdges();
  for (auto& edge : down_streams) {
    if (edge->Dst()->IsSink()) {
      continue;
    } else {
      AddEdge(node->Id(), edge->Dst()->Id());
    }
  }

  for (auto& edge : down_streams) {
    AddNodeRecursive(edge->Dst());
  }
}

void DagProxy::AddNode(DagNodeIdType id, NodeProxy&& node) {
  if (adj_.find(id) == adj_.end()) {
    adj_.insert(std::make_pair(id, std::vector<DagNodeIdType>()));
    nodes_.insert(std::make_pair(id, std::move(node)));
  }
}

void DagProxy::AddEdge(DagNodeIdType s, DagNodeIdType t) {
  adj_[s].push_back(t);

  if (in_degree_.find(t) == in_degree_.end()) {
    in_degree_.insert(std::make_pair(t, 0));
  }
  if (in_degree_.find(s) == in_degree_.end()) {
    in_degree_.insert(std::make_pair(s, 0));
  }
  in_degree_[t]++;
}

}  // namespace act
}  // namespace graphlearn
