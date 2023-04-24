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

#include <memory>
#include <google/protobuf/text_format.h>
#include "common/base/log.h"
#include "core/dag/dag.h"
#include "core/dag/tape.h"
#include "include/dag_request.h"
#include "include/sampling_request.h"
#include "platform/protobuf.h"
#include "generated/proto/service.pb.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class GetDagValuesRequestTest : public ::testing::Test {
public:
  GetDagValuesRequestTest() {
    InitGoogleLogging();
  }
  ~GetDagValuesRequestTest() {
    UninitGoogleLogging();
  }
};

TEST_F(GetDagValuesRequestTest, GetDagValuesRequest) {
  GetDagValuesRequest req(1, 2);
  // In Memory
  {
    EXPECT_EQ(req.Id(), 1);
    EXPECT_EQ(req.ClientId(), 2);
  }
  // RPC
  {
    DagValuesRequestPb pb_req;
    req.SerializeTo(&pb_req);
    GetDagValuesRequest received_req;
    received_req.ParseFrom(&pb_req);
    EXPECT_EQ(received_req.Id(), 1);
    EXPECT_EQ(received_req.ClientId(), 2);
  }
}

TEST_F(GetDagValuesRequestTest, GetDagValuesResponse) {
  // Fill request for serialize
  std::string dag_content =
    "nodes { \n"
      "id: 1 \n"
      "op_name: \"GetNodes\" \n"
    "} \n "
    "nodes { \n"
      "id: 2 \n"
      "op_name: \"FullSampler\" \n"
    "} \n "
    "nodes { \n"
      "id: 3 \n"
      "op_name: \"Sink\" \n"
    "}";
  DagDef def;
  PB_NAMESPACE::TextFormat::ParseFromString(dag_content, &def);
  Dag* dag = nullptr;
  Status s = DagFactory::GetInstance()->Create(def, &dag);
  Tape tape(dag);

  // Record all Dense Respone;
  SamplingResponse res0;
  res0.SetShape(4, 2);
  res0.InitNeighborIds();
  res0.InitEdgeIds();
  for (int64_t i = 0; i < 8; ++i) {
    res0.AppendNeighborId(i + 1);  // kNodeIds
    res0.AppendEdgeId(i);          // kEdgeIds
  }
  // Record from DagNode id = 1.
  // And no need for recording Sink DagNode.
  tape.Record(1, {std::move(res0.tensors_), std::move(res0.sparse_tensors_)});
  
  // // Record Dense-Sparse Response;
  SamplingResponse res1;
  res1.SetShape(4, 6, {1, 2, 3, 4});
  res1.InitNeighborIds();  // kNodeIds
  res1.InitEdgeIds();      // kEdgeIds
  for (int64_t i = 0; i < 10; ++i) {
    res1.AppendNeighborId(i + 1);
    res1.AppendEdgeId(i);
  }
  tape.Record(2, {std::move(res1.tensors_), std::move(res1.sparse_tensors_)});

  GetDagValuesResponse res;
  res.MoveFrom(&tape);
  res.SetEpoch(3);
  res.SetIndex(4);

  // Check Write & Read in Memory;
  {
    EXPECT_EQ(res.Index(), 4);
    EXPECT_EQ(res.Epoch(), 3);
    auto t = res.GetValue(1, kNodeIds);
    auto values = std::get<0>(t);
    ASSERT_TRUE(values != nullptr);
    EXPECT_EQ(values->Size(), 8);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(values->GetInt64(i), i + 1);
    }
    auto indices = std::get<1>(t);
    EXPECT_EQ(indices, nullptr);

    auto t1 = res.GetValue(1, kEdgeIds);
    auto values1 = std::get<0>(t1);
    ASSERT_TRUE(values1 != nullptr);
    EXPECT_EQ(values1->Size(), 8);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(values1->GetInt64(i), i);
    }
    auto indices1 = std::get<1>(t1);
    EXPECT_EQ(indices1, nullptr);

    auto t2= res.GetValue(2, kNodeIds);
    auto values2 = std::get<0>(t2);
    ASSERT_TRUE(values2 != nullptr);
    EXPECT_EQ(values2->Size(), 10);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(values2->GetInt64(i), i + 1);
    }
    auto indices2 = std::get<1>(t2);
    ASSERT_TRUE(indices2 != nullptr);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(indices2->GetInt32(i), i + 1);
    }

    auto t3= res.GetValue(2, kEdgeIds);
    auto values3 = std::get<0>(t3);
    ASSERT_TRUE(values3 != nullptr);
    EXPECT_EQ(values3->Size(), 10);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(values3->GetInt64(i), i);
    }
    auto indices3 = std::get<1>(t3);
    ASSERT_TRUE(indices3 != nullptr);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(indices3->GetInt32(i), i + 1);
    }
  }

  // RPC
  {
    DagValuesResponsePb pb_res;
    res.SerializeTo(&pb_res);
    GetDagValuesResponse received_res;
    received_res.ParseFrom(&pb_res);
    EXPECT_EQ(received_res.Index(), 4);
    EXPECT_EQ(received_res.Epoch(), 3);
    auto t = received_res.GetValue(1, kNodeIds);
    auto values = std::get<0>(t);
    ASSERT_TRUE(values != nullptr);
    EXPECT_EQ(values->Size(), 8);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(values->GetInt64(i), i + 1);
    }
    auto indices = std::get<1>(t);
    EXPECT_EQ(indices, nullptr);

    auto t1 = received_res.GetValue(1, kEdgeIds);
    auto values1 = std::get<0>(t1);
    ASSERT_TRUE(values1 != nullptr);
    EXPECT_EQ(values1->Size(), 8);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(values1->GetInt64(i), i);
    }
    auto indices1 = std::get<1>(t1);
    EXPECT_EQ(indices1, nullptr);

    auto t2= received_res.GetValue(2, kNodeIds);
    auto values2 = std::get<0>(t2);
    ASSERT_TRUE(values2 != nullptr);
    EXPECT_EQ(values2->Size(), 10);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(values2->GetInt64(i), i + 1);
    }
    auto indices2 = std::get<1>(t2);
    ASSERT_TRUE(indices2 != nullptr);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(indices2->GetInt32(i), i + 1);
    }

    auto t3= received_res.GetValue(2, kEdgeIds);
    auto values3 = std::get<0>(t3);
    ASSERT_TRUE(values3 != nullptr);
    EXPECT_EQ(values3->Size(), 10);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(values3->GetInt64(i), i);
    }
    auto indices3 = std::get<1>(t3);
    ASSERT_TRUE(indices3 != nullptr);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(indices3->GetInt32(i), i + 1);
    }
  }
}
