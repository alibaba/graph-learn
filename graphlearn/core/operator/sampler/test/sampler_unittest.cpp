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

#include <unordered_set>

#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/sampling_request.h"
#include "graphlearn/include/graph_request.h"
#include "graphlearn/platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class SamplerTest : public ::testing::Test {
protected:
  void SetUp() override {
    ::graphlearn::io::SideInfo info;
    info.format = ::graphlearn::io::kWeighted;
    info.type = "u-i";
    info.src_type = "user";
    info.dst_type = "item";
    std::unique_ptr<UpdateEdgesRequest> req(new UpdateEdgesRequest(&info, 5));
    std::unique_ptr<UpdateEdgesResponse> res(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value;
    for (int32_t i = 0; i < 5; ++i) {
      GenEdgeValue(&value, i);
      req->Append(&value);
    }
    graph_store_ = new GraphStore(Env::Default());
    Graph* graph = graph_store_->GetGraph("u-i");
    graph->UpdateEdges(req.get(), res.get());
  }

  void TearDown() override {
    delete graph_store_;
  }

  void GenEdgeValue(::graphlearn::io::EdgeValue* value, int32_t index) {
    ::graphlearn::io::IdType src_ids[5] = {0, 0, 0, 1, 1};
    ::graphlearn::io::IdType dst_ids[5] = {10, 20, 30, 11, 21};
    float weights[5] = {1.0, 0.8, 0.5, 1.2, 0.88};
    value->src_id = src_ids[index];
    value->dst_id = dst_ids[index];
    value->weight = weights[index];
  }

protected:
  GraphStore* graph_store_;
};

TEST_F(SamplerTest, Random) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("u-i", "RandomSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  // 1 has neighbors {11, 21}, 2 has no neighbors
  int32_t batch_size = 2;
  int64_t ids[2] = {1, 2};
  req->Set(ids, batch_size);

  OperatorFactory::GetInstance().Set(graph_store_);
  Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  std::unordered_set<int64_t> nbr_set({11, 21});
  const int64_t* neighbor_ids = res->GetNeighborIds();
  // check neighbors of 1
  for (int32_t i = 0; i < nbr_count; ++i) {
    EXPECT_TRUE(nbr_set.find(neighbor_ids[i]) != nbr_set.end());
  }

  // check neighbors of 2, fill with default id
  for (int32_t i = nbr_count; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(neighbor_ids[i] == 0);
  }

  delete res;
  delete req;
}

TEST_F(SamplerTest, Topk) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("u-i", "TopkSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  // 0 has neighbors {10, 20, 30}, 1 has neighbors {11, 21}
  int32_t batch_size = 2;
  int64_t ids[2] = {0, 1};
  req->Set(ids, batch_size);

  OperatorFactory::GetInstance().Set(graph_store_);
  Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  // expected results will be ordered by edge_weight
  int64_t result[4] = {10, 20, 11, 21};
  const int64_t* neighbor_ids = res->GetNeighborIds();
  for (int32_t i = 0; i < batch_size * nbr_count; ++i) {
    EXPECT_EQ(neighbor_ids[i], result[i]);
  }

  delete res;
  delete req;
}

TEST_F(SamplerTest, EdgeWeight) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("u-i", "EdgeWeightSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 2;
  int64_t ids[2] = {0, 1};
  req->Set(ids, batch_size);

  OperatorFactory::GetInstance().Set(graph_store_);
  Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // 0 has neighbors {10, 20, 30}
  std::unordered_set<int64_t> nbr_set_0({10, 20, 30});
  for (int32_t i = 0; i < nbr_count; ++i) {
    EXPECT_TRUE(nbr_set_0.find(neighbor_ids[i]) != nbr_set_0.end());
  }

  // 1 has neighbors {11, 21}
  std::unordered_set<int64_t> nbr_set_1({11, 21});
  for (int32_t i = nbr_count; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(nbr_set_1.find(neighbor_ids[i]) != nbr_set_1.end());
  }

  delete res;
  delete req;
}

TEST_F(SamplerTest, InDegree) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("u-i", "InDegreeSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 2;
  int64_t ids[2] = {0, 1};
  req->Set(ids, batch_size);

  OperatorFactory::GetInstance().Set(graph_store_);
  Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // 0 has neighbors {10, 20, 30}
  std::unordered_set<int64_t> nbr_set_0({10, 20, 30});
  for (int32_t i = 0; i < nbr_count; ++i) {
    EXPECT_TRUE(nbr_set_0.find(neighbor_ids[i]) != nbr_set_0.end());
  }

  // 1 has neighbors {11, 21}
  std::unordered_set<int64_t> nbr_set_1({11, 21});
  for (int32_t i = nbr_count; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(nbr_set_1.find(neighbor_ids[i]) != nbr_set_1.end());
  }

  delete res;
  delete req;
}

TEST_F(SamplerTest, Full) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("u-i", "FullSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 2;
  int64_t ids[2] = {0, 1};
  req->Set(ids, batch_size);

  OperatorFactory::GetInstance().Set(graph_store_);
  Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->IsSparse(), true);

  const int32_t* degrees = res->GetDegrees();
  EXPECT_EQ(degrees[0], 3);
  EXPECT_EQ(degrees[1], 2);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // 0 has neighbors {10, 20, 30}
  std::unordered_set<int64_t> nbr_set_0({10, 20, 30});
  for (int32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(nbr_set_0.find(neighbor_ids[i]) != nbr_set_0.end());
  }

  // 1 has neighbors {11, 21}
  std::unordered_set<int64_t> nbr_set_1({11, 21});
  for (int32_t i = 3; i < 5; ++i) {
    EXPECT_TRUE(nbr_set_1.find(neighbor_ids[i]) != nbr_set_1.end());
  }

  delete res;
  delete req;
}
