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
#include "graphlearn/include/aggregating_request.h"
#include "graphlearn/include/graph_request.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class AggregatingRequestTest : public ::testing::Test {
public:
  AggregatingRequestTest() {
    InitGoogleLogging();
  }
  ~AggregatingRequestTest() {
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

  void CheckFloatAttrs(const float* floats, int32_t size) {
    int32_t batch_size = size / info_.f_num;
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t j = 0; j < info_.f_num; ++j) {
        EXPECT_FLOAT_EQ(floats[i * info_.f_num + j], float(i + j));
      }
    }
  }

  void CheckSegments(const int32_t* segs, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(segs[i], i);
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
        value->attrs->Add(int64_t(index + i));
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        value->attrs->Add(float(index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        value->attrs->Add(std::to_string(index + i));
      }
    }
  }

protected:
  SideInfo info_;
};

TEST_F(AggregatingRequestTest, AggregateNodes) {
  // Fill request for serialize
  AggregatingRequest req("node_type", "SumAggregator");
  int64_t node_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
  req.Set(node_ids, segment_ids, 10, 3);

  // Parse request after deserialize
  EXPECT_EQ(req.Name(), "SumAggregator");
  EXPECT_EQ(req.Type(), "node_type");
  int32_t index = 0;
  int64_t node_id = 0;
  int32_t segment_id = 0;
  while (req.Next(&node_id, &segment_id)) {
    EXPECT_EQ(node_id, index);
    EXPECT_EQ(segment_id, segment_ids[index]);
    ++index;
  }
  EXPECT_EQ(10, index);
  EXPECT_EQ(req.NumSegments(), 3);
  {
    // Fill response for serialize
    info_.format = kAttributed;
    info_.i_num = 10;
    info_.f_num = 15;
    info_.s_num = 1;
    NodeValue value;
    AggregatingResponse res;
    res.SetName("SumAggregator");
    res.SetEmbeddingDim(15);
    res.SetNumSegments(10);
    for (int32_t i = 0; i < 10; ++i) {
      value.attrs->Clear();
      GenNodeValue(&value, i);
      res.AppendEmbedding(value.attrs->GetFloats(nullptr));
      res.AppendSegment(i);
    }

    // Parse response after deserialize
    EXPECT_EQ(res.NumSegments(), 10);
    EXPECT_EQ(res.EmbeddingDim(), 15);

    const float* floats = res.Embeddings();
    CheckFloatAttrs(floats, 15);

    const int32_t* segs = res.Segments();
    CheckSegments(segs, 10);
  }
}
