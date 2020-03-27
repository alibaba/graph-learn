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
#include "graphlearn/include/aggregation_request.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class AggregationRequestTest : public ::testing::Test {
public:
  AggregationRequestTest() {
    InitGoogleLogging();
  }
  ~AggregationRequestTest() {
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

  void CheckUpdateEdges(int32_t reserve_size, int32_t real_size) {
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
    for (int32_t i = 0; i < real_size; ++i) {
      EXPECT_TRUE(req.Next(&value));
      CheckEdgeValue(&value, i);
    }
    EXPECT_TRUE(!req.Next(&value));
  }

  void CheckUpdateNodes(int32_t reserve_size, int32_t real_size) {
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
    for (int32_t i = 0; i < real_size; ++i) {
      EXPECT_TRUE(req.Next(&value));
      CheckNodeValue(&value, i);
    }
    EXPECT_TRUE(!req.Next(&value));
  }

protected:
  SideInfo info_;
};

TEST_F(AggregationRequestTest, AggregateNodes) {
  // Fill request for serialize
  AggregateNodesRequest req("node_type", "SumAggregator");
  int64_t node_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t segments[3] = {3, 4, 3};
  req.Set(node_ids, 10, segments, 3);

  // Parse request after deserialize
  EXPECT_EQ(req.Name(), "SumAggregator");
  EXPECT_EQ(req.NodeType(), "node_type");
  int32_t index = 0;
  int64_t node_id = 0;
  while (req.NextId(&node_id)) {
    EXPECT_EQ(node_id, index);
    ++index;
  }
  EXPECT_EQ(10, index);
  EXPECT_EQ(req.NumSegments(), 3);
  int32_t segment;
  int32_t seg_idx;
  while (req.NextSegment(&segment)) {
    EXPECT_EQ(segments[seg_idx], segment);
    ++seg_idx;
  }

  {
    // Fill response for serialize
    info_.format = kWeighted;
    NodeValue value;
    AggregateNodesResponse res;
    res.SetSideInfo(&info_, 3);
    for (int32_t i = 0; i < 3; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendWeight(value.weight);
    }

    // Parse response after deserialize
    EXPECT_EQ(res.Size(), 3);
    EXPECT_EQ(res.Format(), info_.format);
    const float* weights = res.Weights();
    CheckWeights(weights, 3);
  }
  {
    // Fill response for serialize
    info_.format = kLabeled;
    NodeValue value;
    AggregateNodesResponse res;
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
  }
  {
    // Fill response for serialize
    info_.format = kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    NodeValue value;
    AggregateNodesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
    EXPECT_EQ(res.Size(), 10);
    EXPECT_EQ(res.Format(), info_.format);

    const int64_t* ints = res.IntAttrs();
    const float* floats = res.FloatAttrs();
    const std::string* strings = res.StringAttrs();
    CheckIntAttrs(ints, 10);
    CheckFloatAttrs(floats, 10);
    CheckStringAttrs(strings, 10);
  }
  {
    // Fill response for serialize
    info_.format = kWeighted | kLabeled | kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    NodeValue value;
    AggregateNodesResponse res;
    res.SetSideInfo(&info_, 10);
    for (int32_t i = 0; i < 10; ++i) {
      value.Clear();
      GenNodeValue(&value, i);
      res.AppendWeight(value.weight);
      res.AppendLabel(value.label);
      res.AppendAttribute(&value);
    }

    // Parse response after deserialize
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
  }
}
