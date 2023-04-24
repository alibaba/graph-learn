/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <unordered_map>

#include "core/operator/subgraph/subgraph_sampler.h"
#include "core/operator/subgraph/subgraph_utils.h"

#include "core/operator/op_factory.h"
#include "core/runner/op_runner.h"
#include "platform/env.h"

namespace graphlearn {
namespace op {

Status SubGraphSampler::SampleNeighors(const SamplingRequest* req,
    SamplingResponse* res) {
  Operator* op = OpFactory::GetInstance()->Create("FullSampler");
  std::unique_ptr<OpRunner> runner = GetOpRunner(Env::Default(), op);
  return runner->Run(req, res);
}

Status SubGraphSampler::InduceSubGraph(
    const std::vector<int64_t>& nodes_vec,
    const SubGraphRequest* request,
    SubGraphResponse* response) {
  int32_t node_size = nodes_vec.size();
  SamplingRequest req(request->NbrType(), "FullSampler",
                      GLOBAL_FLAG(DefaultFullNbrNum));
  req.Set(nodes_vec.data(), node_size);
  SamplingResponse res;
  Status s = SampleNeighors(&req, &res);
  RETURN_IF_NOT_OK(s);

  const auto nbrs = res.GetNeighborIds();
  auto& degrees = res.GetShape().segments;
  const auto edge_ids = res.GetEdgeIds();

  response->Init(node_size);
  response->SetNodeIds(nodes_vec.data(), nodes_vec.size());

  int32_t src = 0;
  int32_t dst = 1;
  auto adj_wo_src = subgraph::Graph(node_size);
  auto adj_wo_dst = subgraph::Graph(node_size);

  int32_t offset = 0;
  for (int32_t i = 0; i < node_size; ++i) {
    std::unordered_map<int64_t, int64_t> node2edge;
    node2edge.clear();
    for (int32_t k = offset; k < offset + degrees[i]; ++k) {
      node2edge[nbrs[k]] = edge_ids[k];
    }
    offset += degrees[i];

    for (int32_t j = 0; j < node_size; ++j) {
      auto iter = node2edge.find(nodes_vec[j]);
      if (iter != node2edge.end()) {
        response->AppendEdge(i, j, iter->second);
        response->AppendEdge(j, i, iter->second);
        if (request->NeedDist()) {
          if (i != src && j != src) {
            adj_wo_src.AddEdge(i, j);
            adj_wo_src.AddEdge(j, i);
          }
          if (i != dst && j != dst) {
            adj_wo_dst.AddEdge(i, j);
            adj_wo_dst.AddEdge(j, i);
          }
        }
      }
    }
  }
  if (request->NeedDist()) {
    auto dist_to_dst = adj_wo_src.BFSShortestPath(dst);
    auto dist_to_src = adj_wo_dst.BFSShortestPath(src);
    dist_to_dst[src] = 0;
    dist_to_src[dst] = 0;
    response->SetDistToSrc(dist_to_src.data(), dist_to_src.size());
    response->SetDistToDst(dist_to_dst.data(), dist_to_dst.size());
  }
  return Status::OK();
}

REGISTER_OPERATOR("SubGraphSampler", SubGraphSampler);

}  // namespace op
}  // namespace graphlearn
