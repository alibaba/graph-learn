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

#ifndef GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_
#define GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_

#include <set>
#include <string>

#include "common/base/errors.h"
#include "common/base/macros.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/client.h"
#include "include/config.h"
#include "include/subgraph_request.h"

namespace graphlearn {
namespace op {

class SubGraphSampler : public RemoteOperator {
public:
  virtual ~SubGraphSampler() {}

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const SubGraphRequest* request =
      static_cast<const SubGraphRequest*>(req);
    SubGraphResponse* response =
      static_cast<SubGraphResponse*>(res);

    auto nodes = request->GetSrcIds();
    auto nodes_size = request->BatchSize();
    auto num_nbrs = request->GetNumNbrs();
    auto total_nbr_size = nodes_size;
    auto src_size = nodes_size;
    for (auto num_nbr: num_nbrs) {
      total_nbr_size += src_size * num_nbr;
      src_size *= num_nbr;
    }
    std::vector<int64_t> nodes_vec;
    nodes_vec.reserve(total_nbr_size);
    std::copy(nodes, nodes + nodes_size, std::back_inserter(nodes_vec));

    std::set<int64_t> nbrs_set;
    Status s;
    for (auto num_nbr : num_nbrs) {
      if (num_nbr > 0) {
        SamplingRequest req(request->NbrType(), "FullSampler", num_nbr);
        req.Set(nodes, nodes_size);
        SamplingResponse res;
        s = SampleNeighors(&req, &res);
        RETURN_IF_NOT_OK(s);
        nodes = res.GetNeighborIds();
        int32_t nbrs_count = 0;
        for (int32_t i = 0; i < nodes_size; ++i) {
          nbrs_count += res.GetShape().segments[i];
        }
        nodes_size = nbrs_count;
        for (int32_t i = 0; i < nodes_size; ++i) {
          nbrs_set.insert(nodes[i]);
        }
      }
    }
    std::copy(nbrs_set.begin(), nbrs_set.end(), std::back_inserter(nodes_vec));
    s = InduceSubGraph(nodes_vec, request, response);
    return s;
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }

public:
  virtual Status InduceSubGraph(
      const std::vector<int64_t>& nodes_vec,
      const SubGraphRequest* request,
      SubGraphResponse* response);

protected:
  Status SampleNeighors(const SamplingRequest* req, SamplingResponse* res);

};

}  // namespace op
}  // namespace graphlearn

#endif  //GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_
