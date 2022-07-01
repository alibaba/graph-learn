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

#include <google/protobuf/text_format.h>
#include "gtest/gtest.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/core/operator/op_factory.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/platform/protobuf.h"
#include "graphlearn/proto/dag.pb.h"
#include "graphlearn/core/dag/tape.h"
#include "graphlearn/core/runner/dag_scheduler.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class ThreadDagSchedulerTest : public ::testing::Test {
public:
  ThreadDagSchedulerTest() {
    InitGoogleLogging();
    env_ = Env::Default();
  }

  ~ThreadDagSchedulerTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    // u-i edge: weighted
    ::graphlearn::io::SideInfo info_edge_ui;
    info_edge_ui.format = ::graphlearn::io::kWeighted;
    info_edge_ui.type = "u-i";
    info_edge_ui.src_type = "user";
    info_edge_ui.dst_type = "item";
    std::unique_ptr<UpdateEdgesRequest> req_edge_ui(new UpdateEdgesRequest(&info_edge_ui, 10));
    std::unique_ptr<UpdateEdgesResponse> res_edge_ui(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value_edge_ui;
    for (int32_t i = 0; i < 10; ++i) {
      GenEdgeUIValue(&value_edge_ui, i);
      req_edge_ui->Append(&value_edge_ui);
    }

    // i-i edge: attributed
    ::graphlearn::io::SideInfo info_edge_ii;
    info_edge_ii.format = ::graphlearn::io::kAttributed;
    info_edge_ii.type = "i-i";
    info_edge_ii.src_type = "item";
    info_edge_ii.dst_type = "item";
    info_edge_ii.i_num = 1;
    info_edge_ii.s_num = 1;
    info_edge_ii.f_num = 1;
    std::unique_ptr<UpdateEdgesRequest> req_edge_ii(new UpdateEdgesRequest(&info_edge_ii, 10));
    std::unique_ptr<UpdateEdgesResponse> res_edge_ii(new UpdateEdgesResponse);

    ::graphlearn::io::EdgeValue value_edge_ii;
    for (int32_t i = 0; i < 10; ++i) {
      GenEdgeIIValue(&value_edge_ii, i);
      req_edge_ii->Append(&value_edge_ii);
    }

    // user node: weighted
    ::graphlearn::io::SideInfo info_node_user;
    info_node_user.format = ::graphlearn::io::kWeighted;
    info_node_user.type = "user";
    std::unique_ptr<UpdateNodesRequest> req_node_user(new UpdateNodesRequest(&info_node_user, 10));
    std::unique_ptr<UpdateNodesResponse> res_node_user(new UpdateNodesResponse);

    ::graphlearn::io::NodeValue value_node_user;
    for (int32_t i = 0; i < 10; ++i) {
      GenNodeUserValue(&value_node_user, i);
      req_node_user->Append(&value_node_user);
    }

    // item node: attributed
    ::graphlearn::io::SideInfo info_node_item;
    info_node_item.format = ::graphlearn::io::kAttributed;
    info_node_item.type = "item";
    info_node_item.i_num = 1;
    info_node_item.s_num = 1;
    info_node_item.f_num = 1;
    std::unique_ptr<UpdateNodesRequest> req_node_item(new UpdateNodesRequest(&info_node_item, 10));
    std::unique_ptr<UpdateNodesResponse> res_node_item(new UpdateNodesResponse);

    ::graphlearn::io::NodeValue value_node_item;
    for (int32_t i = 0; i < 10; ++i) {
      GenNodeItemValue(&value_node_item, i);
      req_node_item->Append(&value_node_item);
    }

    graph_store_ = new GraphStore(env_);
    Graph* graph_ui = graph_store_->GetGraph("u-i");
    Graph* graph_ii = graph_store_->GetGraph("i-i");
    Noder* noder_user = graph_store_->GetNoder("user");
    Noder* noder_item = graph_store_->GetNoder("item");
    graph_ui->UpdateEdges(req_edge_ui.get(), res_edge_ui.get());
    graph_ii->UpdateEdges(req_edge_ii.get(), res_edge_ii.get());
    noder_user->UpdateNodes(req_node_user.get(), res_node_user.get());
    noder_item->UpdateNodes(req_node_item.get(), res_node_item.get());

    IndexOption option;
    option.name = "sort";
    graph_ui->Build(option);
    graph_ii->Build(option);
    noder_user->Build(option);
    noder_item->Build(option);

    // Set graph_store.
    // It is need for running operator in DagNode.
    ::graphlearn::op::OpFactory::GetInstance()->Set(graph_store_);
  }

  void TearDown() override {
    delete graph_store_;
  }

  // gen edge ui: weighted.
  // src_id range: (0, 10), dst_id range: (0, 10)
  void GenEdgeUIValue(::graphlearn::io::EdgeValue* value, int32_t index) {
    value->src_id = (int64_t)index;
    value->dst_id = (int64_t)(index + 10);
    value->weight = (float)(index * 0.1);
  }

  // gen edge ii: attributed
  // src_id range: (10, 20), dst_id range: (10, 20)
  void GenEdgeIIValue(::graphlearn::io::EdgeValue* value, int32_t index) {
    int64_t i = (int64_t)(index + 10);
    value->src_id = i;
    value->dst_id = i;
    ::graphlearn::io::AttributeValue* attrs = ::graphlearn::io::NewDataHeldAttributeValue();
    attrs->Reserve(1, 1, 1);
    attrs->Add(i);
    attrs->Add("aloha");
    attrs->Add((float)(i * 0.1));
    value->attrs = attrs;
  }

  // gen node user: weighted.
  // id range: (0, 10)
  void GenNodeUserValue(::graphlearn::io::NodeValue* value, int32_t index) {
    value->id = (int64_t)index;
    value->weight = (float)(index * 0.1);
  }

  // gen node item: attributed.
  // id range: (10, 20)
  void GenNodeItemValue(::graphlearn::io::NodeValue* value, int32_t index) {
    int64_t i = (int64_t)(index + 10);
    value->id = i;
    ::graphlearn::io::AttributeValue* attrs = ::graphlearn::io::NewDataHeldAttributeValue();
    attrs->Reserve(1, 1, 1);
    attrs->Add(i);
    attrs->Add("aloha");
    attrs->Add((float)(i * 0.1));
    value->attrs = attrs;
  }

protected:
  Env* env_;
  GraphStore* graph_store_;
};

TEST_F(ThreadDagSchedulerTest, GetNodes) {
  std::string dag_content =
    "nodes { \n"
      "id: 1 \n"
      "op_name: \"GetNodes\" \n"
      "params { \n"
        "name: \"nf\" \n"
        "length: 1 \n"
        "int32_values: 2 \n"
      "} \n"
      "params { \n"
        "name: \"nt\" \n"
        "dtype: 4 \n"
        "length: 1 \n"
        "string_values: \"user\" \n"
      "} \n"
      "params { \n"
        "name: \"ep\" \n"
        "length: 1 \n"
        "int32_values: 2147483647 \n"
      "} \n"
      "params { \n"
        "name: \"bs\" \n"
        "length: 1 \n"
        "int32_values: 2 \n"
      "} \n"
      "params { \n"
        "name: \"str\" \n"
        "dtype: 4 \n"
        "length: 1 \n"
        "string_values: \"by_order\" \n"
      "} \n"
      "out_edges { \n"
        "id: 1 \n"
        "src_output: \"nid\" \n"
        "dst_input: \"ids\" \n"
      "} \n"
    "} \n"
    "nodes { \n"
      "id: 2 \n"
      "op_name: \"Sink\" \n"
      "in_edges { \n"
        "id: 1 \n"
        "src_output: \"nid\" \n"
        "dst_input: \"ids\" \n"
      "} \n"
    "}";

  DagDef def;
  Dag* dag = nullptr;
  PB_NAMESPACE::TextFormat::ParseFromString(dag_content, &def);
  Status s = DagFactory::GetInstance()->Create(def, &dag);
  EXPECT_TRUE(s.ok());

  SetGlobalFlagEnableActor(0);  // ThreadPoolScheduler
  SetGlobalFlagTapeCapacity(4);

  DagScheduler::Take(env_, dag);
  TapeStorePtr store = GetTapeStore(dag->Id());
  Tape* tape = nullptr;

  // Epoch 0: 0,1 | 2,3 | 4,5 | 6,7 | 8,9 | Fake
  for (int32_t idx = 0; idx < 5; ++idx) {
    tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
    EXPECT_EQ(tape->Id(), idx);
    EXPECT_EQ(tape->Epoch(), 0);
    EXPECT_TRUE(tape->IsReady());
    // Tape record response of 1 DagNode (except SinkNode).
    EXPECT_EQ(tape->Size(), 2);
    // Get Response of DagNode id 1.
    auto& record = tape->Retrieval(1);
    // GetNodes with batch_size 2.
    EXPECT_EQ(record.at("nid").Size(), 2);
  }

  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 5);
  EXPECT_EQ(tape->Epoch(), 0);
  EXPECT_TRUE(tape->IsFaked());

  // Epoch 1: 0,1 | 2,3 | 4,5 | 6,7 | 8,9 | Fake
  for (int32_t idx = 6; idx < 11; ++idx) {
    tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
    EXPECT_EQ(tape->Id(), idx);
    EXPECT_EQ(tape->Epoch(), 1);
    EXPECT_TRUE(tape->IsReady());
    // Tape record response of 1 DagNode (except SinkNode).
    EXPECT_EQ(tape->Size(), 2);
    // Get Response of DagNode id 1.
    auto record = tape->Retrieval(1);
    // GetNodes with batch_size 2.
    EXPECT_EQ(record.at("nid").Size(), 2);
  }

  tape = store->WaitAndPop(GLOBAL_FLAG(ClientId));
  EXPECT_EQ(tape->Id(), 11);
  EXPECT_EQ(tape->Epoch(), 1);
  EXPECT_TRUE(tape->IsFaked());

  env_->SetStopping();

  delete tape;
}
