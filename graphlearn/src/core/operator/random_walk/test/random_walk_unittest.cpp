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

#include <unordered_set>

#include "core/graph/graph_store.h"
#include "core/operator/op_factory.h"
#include "include/random_walk_request.h"
#include "include/graph_request.h"
#include "include/index_option.h"
#include "include/config.h"
#include "include/constants.h"
#include "platform/env.h"
#include "gtest/gtest.h"


using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class RandomWalkTest : public ::testing::Test {
protected:
  void SetUp() override {
    ::graphlearn::io::SideInfo info_edge;
    info_edge.format = ::graphlearn::io::kWeighted;
    info_edge.type = "i-i";
    info_edge.src_type = "i";
    info_edge.dst_type = "i";
    std::unique_ptr<UpdateEdgesRequest> req_edge(new UpdateEdgesRequest(&info_edge, 5));
    std::unique_ptr<UpdateEdgesResponse> res_edge(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value_edge;
    for (int32_t i = 0; i < 10; ++i) {
      GenEdgeValue(&value_edge, i);
      req_edge->Append(&value_edge);
    }

    graph_store_ = new GraphStore(Env::Default());
    Graph* graph = graph_store_->GetGraph("i-i");
    graph->UpdateEdges(req_edge.get(), res_edge.get());

    IndexOption option;
    option.name = "sort";
    graph->Build(option);
  }

  void TearDown() override {
    delete graph_store_;
  }

  void GenEdgeValue(::graphlearn::io::EdgeValue* value, int32_t index) {
    ::graphlearn::io::IdType src_ids[10] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    ::graphlearn::io::IdType dst_ids[10] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    float weights[10] = {0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9};
    value->src_id = src_ids[index];
    value->dst_id = dst_ids[index];
    value->weight = weights[index];
  }

protected:
  GraphStore* graph_store_;
};

TEST_F(RandomWalkTest, WalkOneStep) {
  RandomWalkRequest* req = new RandomWalkRequest("i-i", 1, 1);
  RandomWalkResponse* res = new RandomWalkResponse();

  int32_t batch_size = 2;
  int64_t src_ids[2] = {1, 2};
  req->Set(src_ids, batch_size);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->batch_size_, batch_size);

  for (int32_t i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(res->tensors_[kNodeIds].GetInt64(i) == src_ids[i] ||
                res->tensors_[kNodeIds].GetInt64(i) == src_ids[i] + 1);
  }

  delete res;
  delete req;
}

TEST_F(RandomWalkTest, WalkMultipleSteps) {
  int32_t walk_steps = 3;
  RandomWalkRequest* req = new RandomWalkRequest("i-i", 0.25, 0.25, walk_steps);
  RandomWalkResponse* res = new RandomWalkResponse();

  int32_t batch_size = 2;
  int64_t src_ids[2] = {1, 2};
  int64_t parent_ids[2] = {0, 1};
  int64_t parent_neighbor_ids[4] = {1, 1, 2, 2};
  int32_t parent_neighbor_segments[2] = {2, 2};
  req->Set(src_ids, parent_ids, batch_size, parent_neighbor_ids, parent_neighbor_segments, 4);

  OpFactory::GetInstance()->Set(graph_store_);
  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  Status s = op->Process(req, res);
  EXPECT_TRUE(s.ok());

  EXPECT_EQ(res->batch_size_, batch_size);

  // walks for 1: 1-2-3-4
  EXPECT_EQ(*res->GetWalks(), 2);
  EXPECT_EQ(*(res->GetWalks() + 1), 3);
  EXPECT_EQ(*(res->GetWalks() + 2), 4);
  // walks for 2: 2-3-4-5
  EXPECT_EQ(*(res->GetWalks() + 3), 3);
  EXPECT_EQ(*(res->GetWalks() + 4), 4);
  EXPECT_EQ(*(res->GetWalks() + 5), 5);

  delete res;
  delete req;
}