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

#include <unordered_map>

#include "core/operator/subgraph/subgraph_sampler.h"

#include "include/client.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

Status SampleNeighors(const SamplingRequest* req,
                      SamplingResponse* res) {
  std::unique_ptr<Client> client;
  if (GLOBAL_FLAG(DeployMode) != kLocal) {
    client.reset(NewRpcClient());
  } else {
    client.reset(NewInMemoryClient());
  }
  return client->Sampling(req, res);
}

Status SubGraphSampler::InduceSubGraph(
    const std::set<int64_t>& nodes_set,
    const SubGraphRequest* request,
    SubGraphResponse* response) {
  std::vector<int64_t> nodes(nodes_set.begin(), nodes_set.end());

  int32_t node_size = nodes.size();
  SamplingRequest req(request->NbrType(), "FullSampler", node_size);
  req.Set(nodes.data(), node_size);
  SamplingResponse res;
  Status s = SampleNeighors(&req, &res);
  RETURN_IF_NOT_OK(s);

  const int64_t* nbrs = res.GetNeighborIds();
  const int32_t* degrees = res.GetDegrees();
  const int64_t* edge_ids = res.GetEdgeIds();

  response->Init(node_size);
  response->SetNodeIds(nodes.data(), nodes.size());

  int32_t offset = 0;
  for (int32_t i = 0; i < node_size; ++i) {
    std::unordered_map<int64_t, int64_t> nbr_node_edge_map;
    for (int32_t k = offset; k < offset + degrees[i]; ++k) {
      nbr_node_edge_map[nbrs[k]] = edge_ids[k];
    }
    offset += degrees[i];

    for (int32_t j = 0; j < node_size; ++j) {
      auto iter = nbr_node_edge_map.find(nodes[j]);
      if (iter != nbr_node_edge_map.end()) {
        response->AppendEdge(i, j, iter->second);
      }
    }
  }

  return Status::OK();
}

}  // namespace op
}  // namespace graphlearn
