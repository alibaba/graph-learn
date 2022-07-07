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

#include <cstdlib>
#include <unordered_set>
#include "core/graph/graph_store.h"
#include "core/operator/sampler/sampler.h"
#include "core/operator/op_factory.h"
#include "platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class NegativeSamplerTest : public ::testing::Test {
protected:
  void SetUp() override {
    const int32_t edge_num = 17;
    ::graphlearn::io::SideInfo info;
    info.format = ::graphlearn::io::kWeighted;
    info.type = "u-i";
    info.src_type = "user";
    info.dst_type = "item";
    std::unique_ptr<UpdateEdgesRequest> req(new UpdateEdgesRequest(&info, edge_num));
    std::unique_ptr<UpdateEdgesResponse> res(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value;
    for (int32_t i = 0; i < edge_num; ++i) {
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

  void GenEdgeValue(::graphlearn::io::EdgeValue* value,
                    int32_t index) {
    const int32_t edge_num = 17;
    ::graphlearn::io::IdType src_ids[edge_num] =
      {0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    ::graphlearn::io::IdType dst_ids[edge_num] =
      {10, 20, 30, 11, 21, 31, 10, 20, 10, 1, 2, 3, 4, 5, 6, 7, 8};
    float weights[edge_num] =
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    value->src_id = src_ids[index];
    value->dst_id = dst_ids[index];
    value->weight = weights[index];
  }

protected:
  GraphStore* graph_store_;
};

TEST_F(NegativeSamplerTest, Random) {
  int32_t nbr_count = 5;
  SamplingRequest* req =
    new SamplingRequest("u-i", "RandomNegativeSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 3;
  int64_t ids[3] = {0, 1, 2};
  req->Set(ids, batch_size);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // not restrictly negative, the results exist in the dst ids set
  std::unordered_set<int64_t> neg_set(
    {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21, 30, 31});

  // check negative neighbors
  for (int32_t i = 0; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(neg_set.find(neighbor_ids[i]) != neg_set.end());
  }

  delete res;
  delete req;
}

TEST_F(NegativeSamplerTest, InDegree) {
  int32_t nbr_count = 5;
  SamplingRequest* req =
    new SamplingRequest("u-i", "InDegreeNegativeSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 3;
  int64_t ids[3] = {0, 1, 2};
  req->Set(ids, batch_size);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // check negative neighbors of 0
  std::unordered_set<int64_t> neg_set_0({1, 2, 3, 4, 5, 6, 7, 8, 11, 21, 31});
  for (int32_t i = 0; i < nbr_count; ++i) {
    EXPECT_TRUE(neg_set_0.find(neighbor_ids[i]) != neg_set_0.end());
  }

  // check negative neighbors of 1
  std::unordered_set<int64_t> neg_set_1({1, 2, 3, 4, 5, 6, 7, 8, 30});
  for (int32_t i = nbr_count; i < nbr_count * 2; ++i) {
    EXPECT_TRUE(neg_set_1.find(neighbor_ids[i]) != neg_set_1.end());
  }

  // check negative neighbors of 2
  std::unordered_set<int64_t> neg_set_2({10, 20, 30, 11, 21, 31});
  for (int32_t i = nbr_count * 2; i < nbr_count * 3; ++i) {
    EXPECT_TRUE(neg_set_2.find(neighbor_ids[i]) != neg_set_2.end());
  }

  delete res;
  delete req;
}

TEST_F(NegativeSamplerTest, SoftInDegree) {
  int32_t nbr_count = 5;
  SamplingRequest* req =
    new SamplingRequest("u-i", "SoftInDegreeNegativeSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 3;
  int64_t ids[3] = {0, 1, 2};
  req->Set(ids, batch_size);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // not restrictly negative, the results exist in the dst ids set
  std::unordered_set<int64_t> neg_set(
    {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21, 30, 31});

  // check negative neighbors
  std::unordered_set<int64_t> neg_set_0({1, 2, 3, 4, 5, 6, 7, 8, 11, 21, 31});
  for (int32_t i = 0; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(neg_set.find(neighbor_ids[i]) != neg_set.end());
  }

  delete res;
  delete req;
}
