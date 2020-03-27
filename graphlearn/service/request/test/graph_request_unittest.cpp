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

#include "graphlearn/common/base/log.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/include/graph_request.h"
#include "graphlearn/proto/service.pb.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class GraphRequestTest : public ::testing::Test {
public:
  GraphRequestTest() {
    InitGoogleLogging();
  }
  ~GraphRequestTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    info_.type = "edge_type";
    info_.src_type = "src_type";
    info_.dst_type = "dst_type";
  }

  void TearDown() override {
  }

  void GenEdgeValue(EdgeValue* value, int32_t index) {
    value->src_id = index;
    value->dst_id = index;

    if (info_.IsWeighted()) {
      value->weight = float(index);
    }
    if (info_.IsLabeled()) {
      value->label = index;
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        value->i_attrs.emplace_back(index + i);
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        value->f_attrs.emplace_back(float(index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        value->s_attrs.emplace_back(std::to_string(index + i));
      }
    }
  }

  void CheckWeights(const float* weights, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(weights[i], float(i));
    }
  }

  void CheckLabels(const int32_t* labels, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
      EXPECT_EQ(labels[i], i);
    }
  }

  void CheckIntAttrs(const int64_t* ints, int32_t size) {
    int32_t batch_size = size / info_.i_num;
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t j = 0; j < info_.i_num; ++j) {
        EXPECT_EQ(ints[i * info_.i_num + j], i + j);
      }
    }
  }

  void CheckFloatAttrs(const float* floats, int32_t size) {
    int32_t batch_size = size / info_.f_num;
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t j = 0; j < info_.f_num; ++j) {
        EXPECT_FLOAT_EQ(floats[i * info_.f_num + j], float(i + j));
      }
    }
  }

  void CheckStringAttrs(const std::string* strings, int32_t size) {
    int32_t batch_size = size / info_.s_num;
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t j = 0; j < info_.s_num; ++j) {
        EXPECT_EQ(strings[i * info_.s_num + j], std::to_string(i + j));
      }
    }
  }

  void CheckEdgeValue(EdgeValue* value, int32_t index) {
    EXPECT_EQ(value->src_id, index);
    EXPECT_EQ(value->dst_id, index);

    if (info_.IsLabeled()) {
      EXPECT_EQ(value->label, index);
    }
    if (info_.IsWeighted()) {
      EXPECT_FLOAT_EQ(value->weight, float(index));
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        EXPECT_EQ(value->i_attrs[i], index + i);
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        EXPECT_EQ(value->f_attrs[i], float(index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        EXPECT_EQ(value->s_attrs[i], std::to_string(index + i));
      }
    }
  }

  void GenNodeValue(NodeValue* value, int32_t index) {
    value->id = index;
    if (info_.IsWeighted()) {
      value->weight = float(index);
    }
    if (info_.IsLabeled()) {
      value->label = index;
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        value->i_attrs.emplace_back(index + i);
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        value->f_attrs.emplace_back(float(index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        value->s_attrs.emplace_back(std::to_string(index + i));
      }
    }
  }

  void CheckNodeValue(NodeValue* value, int32_t index) {
    EXPECT_EQ(value->id, index);

    if (info_.IsLabeled()) {
      EXPECT_EQ(value->label, index);
    }
    if (info_.IsWeighted()) {
      EXPECT_FLOAT_EQ(value->weight, float(index));
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        EXPECT_EQ(value->i_attrs[i], index + i);
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        EXPECT_EQ(value->f_attrs[i], float(index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        EXPECT_EQ(value->s_attrs[i], std::to_string(index + i));
      }
    }
  }

  void CheckUpdateEdges(int32_t reserve_size, int32_t real_size, bool use_rpc) {
    // Fill request for serialize
    EdgeValue value;
    UpdateEdgesRequest req(&info_, reserve_size);
    const SideInfo* info = req.GetSideInfo();
    EXPECT_EQ(info->type, info_.type);
    EXPECT_EQ(info->src_type, info_.src_type);
    EXPECT_EQ(info->dst_type, info_.dst_type);

    for (int32_t i = 0; i < real_size; ++i) {
      value.Clear();
      GenEdgeValue(&value, i);
      req.Append(&value);
    }
    EXPECT_EQ(req.Size(), real_size);

    // Parse request after deserialize
    UpdateEdgesRequest* received_req;
    if (use_rpc) {
      OpRequestPb* pb = new OpRequestPb();
      req.SerializeTo(pb);

      received_req = new UpdateEdgesRequest();
      received_req->ParseFrom(pb);
    } else {
      received_req = &req;
    }

    for (int32_t i = 0; i < real_size; ++i) {
      EXPECT_TRUE(received_req->Next(&value));
      CheckEdgeValue(&value, i);
    }
    EXPECT_TRUE(!received_req->Next(&value));

    if (use_rpc) {
      delete received_req;
    }
  }

  void CheckUpdateNodes(int32_t reserve_size, int32_t real_size, bool use_rpc) {
    // Fill request for serialize
    NodeValue value;
    UpdateNodesRequest req(&info_, reserve_size);
    const SideInfo* info = req.GetSideInfo();
    EXPECT_EQ(info->type, info_.type);

    for (int32_t i = 0; i < real_size; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      req.Append(&value);
    }
    EXPECT_EQ(req.Size(), real_size);

    // Parse request after deserialize
    UpdateNodesRequest* received_req;
    if (use_rpc) {
      OpRequestPb* pb = new OpRequestPb();
      req.SerializeTo(pb);

      received_req = new UpdateNodesRequest();
      received_req->ParseFrom(pb);
    } else {
      received_req = &req;
    }

    for (int32_t i = 0; i < real_size; ++i) {
      EXPECT_TRUE(received_req->Next(&value));
      CheckNodeValue(&value, i);
    }
    EXPECT_TRUE(!received_req->Next(&value));

    if (use_rpc) {
      delete received_req;
    }
  }

protected:
  SideInfo info_;
};

TEST_F(GraphRequestTest, UpdateEdges) {
  {
    info_.format = kWeighted;
    CheckUpdateEdges(100, 100, false);
    CheckUpdateEdges(100, 100, true);
  }
  {
    info_.format = kLabeled;
    CheckUpdateEdges(30, 100, false);
    CheckUpdateEdges(30, 100, true);
  }
  {
    info_.format = kAttributed;
    info_.i_num = 10;
    info_.f_num = 3;
    info_.s_num = 0;
    CheckUpdateEdges(80, 100, false);
    CheckUpdateEdges(80, 100, true);
  }
  {
    info_.format = kWeighted | kLabeled | kAttributed;
    info_.i_num = 0;
    info_.f_num = 10;
    info_.s_num = 20;
    CheckUpdateEdges(60, 100, false);
    CheckUpdateEdges(60, 100, true);
  }
}

TEST_F(GraphRequestTest, UpdateNodes) {
  {
    info_.format = kWeighted;
    CheckUpdateNodes(20, 100, false);
    CheckUpdateNodes(20, 100, true);
  }
  {
    info_.format = kLabeled;
    CheckUpdateNodes(30, 100, false);
    CheckUpdateNodes(30, 100, true);
  }
  {
    info_.format = kAttributed;
    info_.i_num = 1;
    info_.f_num = 300;
    info_.s_num = 2;
    CheckUpdateNodes(100, 100, false);
    CheckUpdateNodes(100, 100, true);
  }
  {
    info_.format = kWeighted | kLabeled | kAttributed;
    info_.i_num = 0;
    info_.f_num = 10;
    info_.s_num = 20;
    CheckUpdateNodes(60, 100, false);
    CheckUpdateNodes(60, 100, true);
  }
}

TEST_F(GraphRequestTest, GetEdges) {
  // Fill request for serialize
  GetEdgesRequest req("edge_type", "strategy", 512);

  // Parse request after deserialize
  // In memory
  EXPECT_EQ(req.EdgeType(), "edge_type");
  EXPECT_EQ(req.Strategy(), "strategy");
  EXPECT_EQ(req.BatchSize(), 512);

  // RPC
  OpRequestPb* pb_req = new OpRequestPb();
  req.SerializeTo(pb_req);
  GetEdgesRequest* received_req = new GetEdgesRequest();
  received_req->ParseFrom(pb_req);
  EXPECT_EQ(received_req->Name(), "GetEdges");
  EXPECT_EQ(received_req->EdgeType(), "edge_type");
  EXPECT_EQ(received_req->Strategy(), "strategy");
  EXPECT_EQ(received_req->BatchSize(), 512);

  // Fill response for serialize
  GetEdgesResponse res;
  res.Init(512);
  for (int32_t i = 0; i < 512; ++i) {
    res.Append(i, i + 1, i + 2);
  }

  // Parse response after deserialize
  // In memory
  EXPECT_EQ(res.Size(), 512);
  const int64_t* src_ids = res.SrcIds();
  const int64_t* dst_ids = res.DstIds();
  const int64_t* edge_ids = res.EdgeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(src_ids[i], i);
    EXPECT_EQ(dst_ids[i], i + 1);
    EXPECT_EQ(edge_ids[i], i + 2);
  }
  // RPC
  OpResponsePb* pb_res = new OpResponsePb();
  GetEdgesResponse* received_res = new GetEdgesResponse();
  res.SerializeTo(pb_res);
  received_res->ParseFrom(pb_res);
  EXPECT_EQ(received_res->Size(), 512);
  const int64_t* src_ids_2 = received_res->SrcIds();
  const int64_t* dst_ids_2 = received_res->DstIds();
  const int64_t* edge_ids_2 = received_res->EdgeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(src_ids_2[i], i);
    EXPECT_EQ(dst_ids_2[i], i + 1);
    EXPECT_EQ(edge_ids_2[i], i + 2);
  }
  delete received_res;
}

TEST_F(GraphRequestTest, GetNodesFromNode) {
  // Fill request for serialize
  GetNodesRequest req("node_type", "strategy", NodeFrom::kNode, 512);

  // Parse request after deserialize
  // In memory
  EXPECT_EQ(req.Type(), "node_type");
  EXPECT_EQ(req.Strategy(), "strategy");
  EXPECT_EQ(req.BatchSize(), 512);
  // RPC
  OpRequestPb* pb_req = new OpRequestPb();
  req.SerializeTo(pb_req);
  GetNodesRequest* received_req = new GetNodesRequest();
  received_req->ParseFrom(pb_req);
  EXPECT_EQ(received_req->Name(), "GetNodes");
  EXPECT_EQ(received_req->Type(), "node_type");
  EXPECT_EQ(received_req->Strategy(), "strategy");
  EXPECT_EQ(received_req->GetNodeFrom(), NodeFrom::kNode);
  EXPECT_EQ(received_req->BatchSize(), 512);

  // Fill response for serialize
  GetNodesResponse res;
  res.Init(512);
  for (int32_t i = 0; i < 512; ++i) {
    res.Append(i);
  }

  // Parse response after deserialize
  // In memory
  EXPECT_EQ(res.Size(), 512);
  const int64_t* node_ids = res.NodeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(node_ids[i], i);
  }
  // RPC
  OpResponsePb* pb_res = new OpResponsePb();
  GetNodesResponse* received_res = new GetNodesResponse();
  res.SerializeTo(pb_res);
  received_res->ParseFrom(pb_res);
  EXPECT_EQ(received_res->Size(), 512);
  const int64_t* node_ids_2 = received_res->NodeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(node_ids_2[i], i);
  }
  delete received_res;
}

TEST_F(GraphRequestTest, GetNodesFromEdge) {
  // Fill request for serialize
  GetNodesRequest req("edge_type", "strategy", NodeFrom::kEdgeDst, 512);

  // Parse request after deserialize
  // In memory
  EXPECT_EQ(req.Type(), "edge_type");
  EXPECT_EQ(req.Strategy(), "strategy");
  EXPECT_EQ(req.GetNodeFrom(), NodeFrom::kEdgeDst);
  EXPECT_EQ(req.BatchSize(), 512);
  // RPC
  OpRequestPb* pb_req = new OpRequestPb();
  req.SerializeTo(pb_req);
  GetNodesRequest* received_req = new GetNodesRequest();
  received_req->ParseFrom(pb_req);
  EXPECT_EQ(received_req->Name(), "GetNodes");
  EXPECT_EQ(received_req->Type(), "edge_type");
  EXPECT_EQ(received_req->Strategy(), "strategy");
  EXPECT_EQ(received_req->GetNodeFrom(), NodeFrom::kEdgeDst);
  EXPECT_EQ(received_req->BatchSize(), 512);

  // Fill response for serialize
  GetNodesResponse res;
  res.Init(512);
  for (int32_t i = 0; i < 512; ++i) {
    res.Append(i);
  }

  // Parse response after deserialize
  // In memory
  EXPECT_EQ(res.Size(), 512);
  const int64_t* node_ids = res.NodeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(node_ids[i], i);
  }
  // RPC
  OpResponsePb* pb_res = new OpResponsePb();
  GetNodesResponse* received_res = new GetNodesResponse();
  res.SerializeTo(pb_res);
  received_res->ParseFrom(pb_res);
  EXPECT_EQ(received_res->Size(), 512);
  const int64_t* node_ids_2 = received_res->NodeIds();
  for (int32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(node_ids_2[i], i);
  }
  delete received_res;
}

TEST_F(GraphRequestTest, LookupEdges) {
  // Fill request for serialize
  LookupEdgesRequest req("edge_type");
  int64_t edge_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int64_t src_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(edge_ids, src_ids, 10);

  // Parse request after deserialize
  // In memory
  {
    EXPECT_EQ(req.Name(), "LookupEdges");
    EXPECT_EQ(req.EdgeType(), "edge_type");
    int32_t index = 0;
    int64_t edge_id, src_id;
    while (req.Next(&edge_id, &src_id)) {
      EXPECT_EQ(edge_id, index);
      EXPECT_EQ(src_id, index);
      ++index;
    }
    EXPECT_EQ(10, index);
  }
  // RPC
  {
    OpRequestPb* pb_req = new OpRequestPb();
    req.SerializeTo(pb_req);
    LookupEdgesRequest* received_req = new LookupEdgesRequest();
    received_req->ParseFrom(pb_req);
    EXPECT_EQ(received_req->Name(), "LookupEdges");
    EXPECT_EQ(received_req->EdgeType(), "edge_type");
    int32_t index = 0;
    int64_t edge_id, src_id;
    while (received_req->Next(&edge_id, &src_id)) {
      EXPECT_EQ(edge_id, index);
      EXPECT_EQ(src_id, index);
      ++index;
    }
    EXPECT_EQ(10, index);
    delete received_req;
  }

  {
    // Fill response for serialize
    info_.format = kWeighted;
    EdgeValue value;
    LookupEdgesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenEdgeValue(&value, i);
      res.AppendWeight(value.weight);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const float* weights = res.Weights();
    CheckWeights(weights, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupEdgesResponse* received_res = new LookupEdgesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    weights = received_res->Weights();
    CheckWeights(weights, 10);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kLabeled;
    EdgeValue value;
    LookupEdgesResponse res;
    res.SetSideInfo(&info_, 20);
    for (int32_t i = 0; i < 20; ++i) {
      value.Clear();
      GenEdgeValue(&value, i);
      res.AppendLabel(value.label);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 20);
    EXPECT_EQ(res.Format(), info_.format);
    const int32_t* labels = res.Labels();
    CheckLabels(labels, 20);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupEdgesResponse* received_res = new LookupEdgesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 20);
    EXPECT_EQ(received_res->Format(), info_.format);
    labels = received_res->Labels();
    CheckLabels(labels, 20);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    EdgeValue value;
    LookupEdgesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenEdgeValue(&value, i);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const int64_t* ints = res.IntAttrs();
    const float* floats = res.FloatAttrs();
    const std::string* strings = res.StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupEdgesResponse* received_res = new LookupEdgesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    ints = received_res->IntAttrs();
    floats = received_res->FloatAttrs();
    strings = received_res->StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kWeighted | kLabeled | kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    EdgeValue value;
    LookupEdgesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenEdgeValue(&value, i);
      res.AppendWeight(value.weight);
      res.AppendLabel(value.label);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const int32_t* labels = res.Labels();
    CheckLabels(labels, 10);
    const float* weights = res.Weights();
    CheckWeights(weights, 10);
    const int64_t* ints = res.IntAttrs();
    const float* floats = res.FloatAttrs();
    const std::string* strings = res.StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupEdgesResponse* received_res = new LookupEdgesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    labels = received_res->Labels();
    CheckLabels(labels, 10);
    weights = received_res->Weights();
    CheckWeights(weights, 10);
    ints = received_res->IntAttrs();
    floats = received_res->FloatAttrs();
    strings = received_res->StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    delete received_res;
  }
}

TEST_F(GraphRequestTest, LookupNodes) {
  // Fill request for serialize
  LookupNodesRequest req("node_type");
  int64_t node_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(node_ids, 10);

  // Parse request after deserialize
  // In memory
  {
    EXPECT_EQ(req.Name(), "LookupNodes");
    EXPECT_EQ(req.NodeType(), "node_type");
    int32_t index = 0;
    int64_t node_id = 0;
    while (req.Next(&node_id)) {
      EXPECT_EQ(node_id, index);
      ++index;
    }
    EXPECT_EQ(10, index);
  }
  // RPC
  {
    OpRequestPb* pb_req = new OpRequestPb();
    req.SerializeTo(pb_req);
    LookupNodesRequest* received_req = new LookupNodesRequest();
    received_req->ParseFrom(pb_req);
    EXPECT_EQ(received_req->Name(), "LookupNodes");
    EXPECT_EQ(received_req->NodeType(), "node_type");
    int32_t index = 0;
    int64_t node_id = 0;
    while (received_req->Next(&node_id)) {
      EXPECT_EQ(node_id, index);
      ++index;
    }
    EXPECT_EQ(10, index);
    delete received_req;
  }

  {
    // Fill response for serialize
    info_.format = kWeighted;
    NodeValue value;
    LookupNodesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendWeight(value.weight);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const float* weights = res.Weights();
    CheckWeights(weights, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupNodesResponse* received_res = new LookupNodesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    weights = received_res->Weights();
    CheckWeights(weights, 10);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kLabeled;
    NodeValue value;
    LookupNodesResponse res;
    res.SetSideInfo(&info_, 20);
    for (int32_t i = 0; i < 20; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendLabel(value.label);
    }

    // Parse response after deserialize
    EXPECT_EQ(res.Size(), 20);
    EXPECT_EQ(res.Format(), info_.format);
    const int32_t* labels = res.Labels();
    CheckLabels(labels, 20);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupNodesResponse* received_res = new LookupNodesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 20);
    EXPECT_EQ(received_res->Format(), info_.format);
    labels = received_res->Labels();
    CheckLabels(labels, 20);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    NodeValue value;
    LookupNodesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const int64_t* ints = res.IntAttrs();
    const float* floats = res.FloatAttrs();
    const std::string* strings = res.StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupNodesResponse* received_res = new LookupNodesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    ints = received_res->IntAttrs();
    floats = received_res->FloatAttrs();
    strings = received_res->StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    delete received_res;
  }
  {
    // Fill response for serialize
    info_.format = kWeighted | kLabeled | kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    NodeValue value;
    LookupNodesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendWeight(value.weight);
      res.AppendLabel(value.label);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);
    const int32_t* labels = res.Labels();
    CheckLabels(labels, 10);
    const float* weights = res.Weights();
    CheckWeights(weights, 10);
    const int64_t* ints = res.IntAttrs();
    const float* floats = res.FloatAttrs();
    const std::string* strings = res.StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    LookupNodesResponse* received_res = new LookupNodesResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->Size(), 10);
    EXPECT_EQ(received_res->Format(), info_.format);
    labels = received_res->Labels();
    CheckLabels(labels, 10);
    weights = received_res->Weights();
    CheckWeights(weights, 10);
    ints = received_res->IntAttrs();
    floats = received_res->FloatAttrs();
    strings = received_res->StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
    delete received_res;
  }
}
