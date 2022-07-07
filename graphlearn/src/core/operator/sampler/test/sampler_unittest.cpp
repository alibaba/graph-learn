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

#include "core/graph/graph_store.h"
#include "core/operator/sampler/sampler.h"
#include "core/operator/op_factory.h"
#include "include/sampling_request.h"
#include "include/graph_request.h"
#include "include/index_option.h"
#include "platform/env.h"
#include "gtest/gtest.h"
#include "include/config.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class SamplerTest : public ::testing::Test {
protected:
  void SetUp() override {
    ::graphlearn::io::SideInfo info_edge;
    info_edge.format = ::graphlearn::io::kWeighted;
    info_edge.type = "u-i";
    info_edge.src_type = "user";
    info_edge.dst_type = "item";
    std::unique_ptr<UpdateEdgesRequest> req_edge(new UpdateEdgesRequest(&info_edge, 5));
    std::unique_ptr<UpdateEdgesResponse> res_edge(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value_edge;
    for (int32_t i = 0; i < 5; ++i) {
      GenEdgeValue(&value_edge, i);
      req_edge->Append(&value_edge);
    }

    ::graphlearn::io::SideInfo info_node;
    info_node.format = ::graphlearn::io::kWeighted;
    info_node.type = "user";
    std::unique_ptr<UpdateNodesRequest> req_node(new UpdateNodesRequest(&info_node, 5));
    std::unique_ptr<UpdateNodesResponse> res_node(new UpdateNodesResponse);

    ::graphlearn::io::NodeValue value_node;
    for (int32_t i = 0; i < 5; ++i) {
      GenNodeValue(&value_node, i);
      req_node->Append(&value_node);
    }

    graph_store_ = new GraphStore(Env::Default());
    Graph* graph = graph_store_->GetGraph("u-i");
    Noder* noder = graph_store_->GetNoder("user");
    graph->UpdateEdges(req_edge.get(), res_edge.get());
    noder->UpdateNodes(req_node.get(), res_node.get());

    IndexOption option;
    option.name = "sort";
    graph->Build(option);
    noder->Build(option);
  }

  void TearDown() override {
    delete graph_store_;
  }

  void GenEdgeValue(::graphlearn::io::EdgeValue* value, int32_t index) {
    ::graphlearn::io::IdType src_ids[5] = {0, 0, 0, 1, 1};
    ::graphlearn::io::IdType dst_ids[5] = {10, 20, 30, 11, 21};
    float weights[5] = {0.8, 1.0, 0.5, 0.88, 1.2};
    value->src_id = src_ids[index];
    value->dst_id = dst_ids[index];
    value->weight = weights[index];
  }

  void GenNodeValue(::graphlearn::io::NodeValue* value, int32_t index) {
    ::graphlearn::io::IdType node_ids[5] = {0, 1, 2, 3, 4};
    float weights[5] = {0.8, 1.0, 0.5, 0.88, 1.2};
    value->id = node_ids[index];
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

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
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

TEST_F(SamplerTest, RandomWithoutReplacement) {
  int32_t nbr_count = 3;
  SamplingRequest* req = new SamplingRequest(
      "u-i", "RandomWithoutReplacementSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  // 1 has neighbors {11, 21}, 2 has no neighbors
  int32_t batch_size = 2;
  int64_t ids[2] = {1, 2};
  req->Set(ids, batch_size);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
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

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->NeighborCount(), nbr_count);
  EXPECT_EQ(res->IsSparse(), false);

  // expected results will be ordered by edge_weight
  int64_t result[4] = {20, 10, 21, 11};
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

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
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

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
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

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->BatchSize(), batch_size);
  EXPECT_EQ(res->IsSparse(), true);

  const int32_t* degrees = res->GetDegrees();
  EXPECT_EQ(degrees[0], 2);
  EXPECT_EQ(degrees[1], 2);

  const int64_t* neighbor_ids = res->GetNeighborIds();

  // 0 has neighbors {10, 20, 30}
  std::unordered_set<int64_t> nbr_set_0({10, 20, 30});
  for (int32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(nbr_set_0.find(neighbor_ids[i]) != nbr_set_0.end());
  }

  // 1 has neighbors {11, 21}
  std::unordered_set<int64_t> nbr_set_1({11, 21});
  for (int32_t i = 2; i < 4; ++i) {
    EXPECT_TRUE(nbr_set_1.find(neighbor_ids[i]) != nbr_set_1.end());
  }

  delete res;
  delete req;
}

TEST_F(SamplerTest, DISABLED_NodeWeightNegative) {
  int32_t nbr_count = 2;
  SamplingRequest* req = new SamplingRequest("user", "NodeWeightNegativeSampler", nbr_count);
  SamplingResponse* res = new SamplingResponse();

  int32_t batch_size = 2;
  int64_t ids[2] = {0, 1};
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

  // res should not be in neg_set
  std::unordered_set<int64_t> neg_set({0, 1});
  // res should be in pos_set
  std::unordered_set<int64_t> pos_set({2, 3, 4});
  for (int32_t i = 0; i < batch_size * nbr_count; ++i) {
    EXPECT_TRUE(neg_set.find(neighbor_ids[i]) == neg_set.end());
    EXPECT_TRUE(pos_set.find(neighbor_ids[i]) != pos_set.end());
  }

  delete res;
  delete req;
}
