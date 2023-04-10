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

#include "common/base/log.h"
#include "include/sampling_request.h"
#include "include/graph_request.h"
#include "generated/proto/service.pb.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class SamplingRequestTest : public ::testing::Test {
public:
  SamplingRequestTest() {
    InitGoogleLogging();
  }
  ~SamplingRequestTest() {
    UninitGoogleLogging();
  }
};

TEST_F(SamplingRequestTest, DenseShape) {
  // Fill request for serialize

  SamplingRequest req("i-i", "RandomSampler", 3);
  int64_t src_ids[4] = {1, 2, 3, 4};
  req.Set(src_ids, 4);

  // Parse request after deserialize
  // In memory
  {
    EXPECT_EQ(req.Name(), "RandomSampler");
    EXPECT_EQ(req.Type(), "i-i");
    EXPECT_EQ(req.Strategy(), "RandomSampler");
    EXPECT_EQ(req.BatchSize(), 4);
    EXPECT_EQ(req.NeighborCount(), 3);

    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(req.GetSrcIds() + i), src_ids[i]);
    }
  }
  // RPC
  {
    OpRequestPb* pb_req = new OpRequestPb();
    req.SerializeTo(pb_req);
    SamplingRequest* received_req = new SamplingRequest();
    received_req->ParseFrom(pb_req);
    EXPECT_EQ(received_req->Name(), "RandomSampler");
    EXPECT_EQ(received_req->Type(), "i-i");
    EXPECT_EQ(received_req->Strategy(), "RandomSampler");
    EXPECT_EQ(received_req->BatchSize(), 4);
    EXPECT_EQ(received_req->NeighborCount(), 3);

    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(received_req->GetSrcIds() + i), src_ids[i]);
    }
    delete received_req;
  }

  {
    // Fill response for serialize
    SamplingResponse res;
    res.SetShape(4, 2);
    res.InitNeighborIds();
    res.InitEdgeIds();

    int64_t nbrs[8];
    for (int64_t i = 0; i < 8; ++i) {
      nbrs[i] = i + 1;
      res.AppendNeighborId(i + 1);
      res.AppendEdgeId(i);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.GetShape().dim1, 4);
    EXPECT_EQ(res.GetShape().dim2, 2);
    EXPECT_EQ(res.GetShape().size, 8);
    for (int64_t i = 0; i < 8; ++i) {
      EXPECT_EQ(*(res.GetNeighborIds() + i), i + 1);
      EXPECT_EQ(*(res.GetEdgeIds() + i), i);
    }

    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    SamplingResponse* received_res = new SamplingResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->GetShape().dim1, 4);
    EXPECT_EQ(received_res->GetShape().dim2, 2);
    EXPECT_EQ(received_res->GetShape().size, 8);
    for (int64_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(received_res->GetNeighborIds() + i), i + 1);
      EXPECT_EQ(*(received_res->GetEdgeIds() + i), i);
    }
    delete received_res;
  }
}

TEST_F(SamplingRequestTest, SparseShape) {
  // Fill request for serialize

  SamplingRequest req("i-i", "FullSampler", 3);
  int64_t src_ids[4] = {1, 2, 3, 4};
  req.Set(src_ids, 4);

  // Parse request after deserialize
  // In memory
  {
    EXPECT_EQ(req.Name(), "FullSampler");
    EXPECT_EQ(req.Type(), "i-i");
    EXPECT_EQ(req.Strategy(), "FullSampler");
    EXPECT_EQ(req.BatchSize(), 4);
    EXPECT_EQ(req.NeighborCount(), 3);

    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(req.GetSrcIds() + i), src_ids[i]);
    }
  }
  // RPC
  {
    OpRequestPb* pb_req = new OpRequestPb();
    req.SerializeTo(pb_req);
    SamplingRequest* received_req = new SamplingRequest();
    received_req->ParseFrom(pb_req);
    EXPECT_EQ(received_req->Name(), "FullSampler");
    EXPECT_EQ(received_req->Type(), "i-i");
    EXPECT_EQ(received_req->Strategy(), "FullSampler");
    EXPECT_EQ(received_req->BatchSize(), 4);
    EXPECT_EQ(received_req->NeighborCount(), 3);

    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(received_req->GetSrcIds() + i), src_ids[i]);
    }
    delete received_req;
  }

  {
    // Fill response for serialize
    SamplingResponse res;
    res.SetShape(4, 6, {1, 2, 3, 4});
    res.InitNeighborIds();
    res.InitEdgeIds();

    for (int64_t i = 0; i < 10; ++i) {
      res.AppendNeighborId(i + 1);
      res.AppendEdgeId(i);
    }

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.GetShape().dim1, 4);
    EXPECT_EQ(res.GetShape().dim2, 6);
    EXPECT_EQ(res.GetShape().size, 10);
    std::vector<int32_t> segments{1, 2, 3, 4};
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(res.GetShape().segments[i], segments[i]);
    }
    for (int64_t i = 0; i < 10; ++i) {
      EXPECT_EQ(*(res.GetNeighborIds() + i), i + 1);
      EXPECT_EQ(*(res.GetEdgeIds() + i), i);
    }

    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    SamplingResponse* received_res = new SamplingResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->GetShape().dim1, 4);
    EXPECT_EQ(received_res->GetShape().dim2, 6);
    EXPECT_EQ(received_res->GetShape().size, 10);
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(received_res->GetShape().segments[i], segments[i]);
    }
    for (int64_t i = 0; i < 10; ++i) {
      EXPECT_EQ(*(received_res->GetNeighborIds() + i), i + 1);
      EXPECT_EQ(*(received_res->GetEdgeIds() + i), i);
    }
    delete received_res;
  }
}