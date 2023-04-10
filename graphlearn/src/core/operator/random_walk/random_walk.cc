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

#include <cmath>
#include <random>
#include "include/random_walk_request.h"
#include "core/operator/op_registry.h"
#include "core/operator/operator.h"
#include "core/operator/sampler/alias_method.h"
#include "core/runner/op_runner.h"
#include "include/client.h"
#include "include/config.h"
#include "include/constants.h"


namespace graphlearn {
namespace op {

class RandomWalk : public RemoteOperator {
public:
  virtual ~RandomWalk() {}

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const RandomWalkRequest* request =
      static_cast<const RandomWalkRequest*>(req);
    RandomWalkResponse* response =
      static_cast<RandomWalkResponse*>(res);

    Status s;
    auto walk_len = request->WalkLen();
    auto batch_size = request->BatchSize();

    size_t total_size = batch_size * walk_len;
    response->SetBatchSize(batch_size);
    response->InitWalks(total_size);

    std::vector<int64_t> ret_walks;
    ret_walks.reserve(total_size);

    std::unique_ptr<OpRunner> runner = GetOpRunner(Env::Default(), this);

    if (request->IsDeepWalk()) {
      // Walk the 1st steps locally.
      this->DeepWalk(request, &ret_walks);
      if (walk_len <= 1) {
        response->AppendWalks(ret_walks.data(), total_size);
        return s;
      }

      // Walk the 2-N steps remote.
      for (int32_t step = 1; step < walk_len; ++step) {
        RandomWalkRequest sub_req(request->Type(), request->P(), request->Q());
        RandomWalkResponse sub_res;

        std::vector<int64_t> sub_walks;
        sub_walks.reserve(batch_size);
        auto begin = ret_walks.begin() + batch_size * (step - 1);
        std::copy(begin, begin + batch_size, std::back_inserter(sub_walks));
        sub_req.Set(sub_walks.data(), sub_walks.size());

        s = runner->Run(&sub_req, &sub_res);
        if (!s.ok()) return s;
        std::copy(sub_res.GetWalks(), sub_res.GetWalks() + batch_size, std::back_inserter(ret_walks));
      }
    } else {
      std::vector<int64_t> ret_nbrs;
      std::vector<int32_t> ret_degrees;
      int32_t total_neighbor_count = 0; // Total neighbor count for the batch src_ids.
      ret_nbrs.reserve(GLOBAL_FLAG(DefaultFullNbrNum) * batch_size);
      ret_degrees.reserve(batch_size);

      this->WeightedRandomWalk(request, &ret_walks, &ret_nbrs, &ret_degrees);

      for (int32_t i = 0; i < batch_size; ++i) {
        total_neighbor_count += ret_degrees[i];
      }

      if (walk_len <= 1) {
        response->AppendWalks(ret_walks.data(), total_size);
        response->InitNeighbors(batch_size, total_neighbor_count);
        response->AppendNeighborIds(ret_nbrs.data(), total_neighbor_count);
        response->AppendDegrees(ret_degrees.data(), batch_size);
        return s;
      }

      for (int32_t step = 1; step < walk_len; ++step) {
        RandomWalkRequest sub_req(request->Type(), request->P(), request->Q());
        RandomWalkResponse sub_res;

        std::vector<int64_t> sub_walks;
        sub_walks.reserve(batch_size);
        std::copy(ret_walks.begin() + batch_size * (step - 1), ret_walks.begin() + batch_size * step,
                  std::back_inserter(sub_walks));
        std::vector<int64_t> sub_src;
        sub_src.reserve(batch_size);
        if (step == 1) {
          std::copy(request->GetSrcIds(), request->GetSrcIds() + batch_size, std::back_inserter(sub_src));
        } else {
          auto begin = ret_walks.begin() + batch_size * (step - 2);
          std::copy(begin, begin + batch_size, std::back_inserter(sub_src));
        }

        sub_req.Set(sub_walks.data(),
                    sub_src.data(),
                    batch_size,
                    ret_nbrs.data(),
                    ret_degrees.data(),
                    total_neighbor_count);
        s = runner->Run(&sub_req, &sub_res);
        if (!s.ok()) return s;

        auto degrees = sub_res.GetDegrees();
        total_neighbor_count = 0;
        for (int32_t i = 0; i < batch_size; ++i) {
          total_neighbor_count += *(degrees + i);
        }
        std::copy(sub_res.GetWalks(), sub_res.GetWalks() + batch_size, std::back_inserter(ret_walks));
        ret_nbrs.clear();
        ret_degrees.clear();
        std::copy(sub_res.GetNeighborIds(), sub_res.GetNeighborIds() + total_neighbor_count,  std::back_inserter(ret_nbrs));
        std::copy(degrees, degrees + batch_size, std::back_inserter(ret_degrees));
      }

    }

    int64_t ret[total_size];
    for (size_t row = 0; row < walk_len; ++row) {
      for (size_t col = 0; col < batch_size; ++col) {
        ret[col * walk_len + row] = ret_walks[row * batch_size + col];
      }
    }

    response->AppendWalks(ret, total_size);
    return s;
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    const RandomWalkRequest* request =
      static_cast<const RandomWalkRequest*>(req);
    RandomWalkResponse* response =
      static_cast<RandomWalkResponse*>(res);
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->RandomWalk(request, response);
  }

private:
void DeepWalk(const RandomWalkRequest* req,
             std::vector<int64_t>* ret_walks) {
  int32_t batch_size = req->BatchSize();
  auto src_ids = req->GetSrcIds();

  const std::string& edge_type = req->Type();
  Graph* graph = graph_store_->GetGraph(edge_type);
  auto storage = graph->GetLocalStorage();

  for (int32_t i = 0; i < batch_size; ++i) {
      auto src_id = src_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);

      auto nbr_count = neighbor_ids.Size();
      if (nbr_count == 0) {
        ret_walks->push_back(GLOBAL_FLAG(DefaultNeighborId));
      } else {
        thread_local static std::random_device rd;
        thread_local static std::mt19937 engine(rd());
        std::uniform_int_distribution<> dist(0, nbr_count - 1);
        int32_t indice = dist(engine);
        ret_walks->push_back(neighbor_ids[indice]);
      }
    }
  }

  void WeightedRandomWalk(const RandomWalkRequest* req,
                          std::vector<int64_t>* ret_walks,
                          std::vector<int64_t>* ret_nbrs,
                          std::vector<int32_t>* ret_degrees) {
    int32_t batch_size = req->BatchSize();
    auto src_ids = req->GetSrcIds();

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    auto parent_ids = req->GetParentIds();
    auto parent_neighbor_ids = req->GetParentNeighborIds();
    auto parent_neighbor_segments = req->GetParentNeighborSegments();

    float biased_weights[GLOBAL_FLAG(DefaultFullNbrNum)];
    int32_t cursor = 0;

    for (int32_t i = 0; i < batch_size; ++i) {
      auto src_id = src_ids[i];
      auto parent_id = parent_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);

      auto nbr_count = neighbor_ids.Size();
      if (nbr_count == 0) {
        ret_walks->push_back(GLOBAL_FLAG(DefaultNeighborId));
      } else {
        auto indice = WeightedRandomWalkKernel(
          src_id, parent_id, parent_neighbor_ids,
          cursor, parent_neighbor_segments[i],
          req->P(), req->Q(), storage, biased_weights);
        cursor += parent_neighbor_segments[i];

        ret_walks->push_back(neighbor_ids[indice]);
      }

      int32_t degree = std::min(nbr_count, GLOBAL_FLAG(DefaultFullNbrNum));
      if (degree > 0) {
        std::copy_n(neighbor_ids.data(), degree, std::back_inserter(*ret_nbrs));
      }
      ret_degrees->push_back(degree);
    }
  }

  int32_t WeightedRandomWalkKernel(int64_t src_id,
                                   int64_t parent_id,
                                   const int64_t* parent_neighbor_ids,
                                   int32_t start,
                                   int32_t size,
                                   float p,
                                   float q,
                                   io::GraphStorage* storage,
                                   float* biased_weights) {
    auto neighbor_ids = storage->GetNeighbors(src_id);
    auto edge_ids = storage->GetOutEdges(src_id);

    size_t nbr_count = std::min(edge_ids.Size(), GLOBAL_FLAG(DefaultFullNbrNum));

    for (size_t nbr_idx = 0; nbr_idx < nbr_count; ++nbr_idx) {
      auto edge_weight = storage->GetEdgeWeight(edge_ids[nbr_idx]);
      if (neighbor_ids[nbr_idx] == parent_id) {
        biased_weights[nbr_idx] = edge_weight * 1.0 / (p + 1e-6);
      } else {
        int32_t idx = start;
        for (; idx < start + size; ++idx) {
          if (parent_neighbor_ids[idx] == neighbor_ids[nbr_idx]) {
            biased_weights[nbr_idx] = edge_weight;
            break;
          }
        }
        if (idx == start + size) {
          biased_weights[nbr_idx] = edge_weight * 1.0 / (q + 1e-6);
        }
      }
    }

    std::vector<float> weights(biased_weights, biased_weights + nbr_count);

    AliasMethod am(&weights);
    int32_t indice;
    am.Sample(1, &indice);
    return indice;
  }
};

REGISTER_OPERATOR("RandomWalk", RandomWalk);

}  // namespace op
}  // namespace graphlearn
