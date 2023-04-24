/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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
#include "include/random_walk_request.h"
#include "include/graph_request.h"
#include "generated/proto/service.pb.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class RandomWalkRequestTest : public ::testing::Test {
public:
  RandomWalkRequestTest() {
    InitGoogleLogging();
  }
  ~RandomWalkRequestTest() {
    UninitGoogleLogging();
  }
};

TEST_F(RandomWalkRequestTest, RandomWalk) {
  // Fill request for serialize
  float p = 0.25;
  float q = 0.25;

  RandomWalkRequest req("i-i", p, q, 3);
  int64_t src_ids[4] = {1, 2, 3, 4};
  int64_t parent_ids[4] = {0, 1, 2, 3};
  int64_t parent_neighbor_ids[6] = {1, 2, 2, 3, 3, 3};
  int32_t parent_neighbor_segments[4] = {0, 1, 2, 3};
  req.Set(src_ids, parent_ids, 4, parent_neighbor_ids, parent_neighbor_segments, 6);

  // Parse request after deserialize
  // In memory
  {
    EXPECT_EQ(req.Name(), "RandomWalk");
    EXPECT_EQ(req.Type(), "i-i");
    EXPECT_FLOAT_EQ(req.P(), p);
    EXPECT_FLOAT_EQ(req.Q(), q);
    EXPECT_EQ(req.BatchSize(), 4);
    EXPECT_EQ(req.WalkLen(), 3);

    int32_t cursor = 0;
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(req.GetSrcIds() + i), src_ids[i]);
      EXPECT_EQ(*(req.GetParentIds() + i), parent_ids[i]);
      EXPECT_EQ(*(req.GetParentNeighborSegments() + i), parent_ids[i]);
      for (int32_t j = 0; j < i; ++j) {
        EXPECT_EQ(*(req.GetParentNeighborIds() + cursor + j), parent_ids[i]);
      }
      cursor += *(req.GetParentNeighborSegments() + i);
    }
  }
  // RPC
  {
    OpRequestPb* pb_req = new OpRequestPb();
    req.SerializeTo(pb_req);
    RandomWalkRequest* received_req = new RandomWalkRequest();
    received_req->ParseFrom(pb_req);
    EXPECT_EQ(received_req->Name(), "RandomWalk");
    EXPECT_EQ(received_req->Type(), "i-i");
    EXPECT_FLOAT_EQ(received_req->P(), p);
    EXPECT_FLOAT_EQ(received_req->Q(), q);
    EXPECT_EQ(received_req->BatchSize(), 4);
    EXPECT_EQ(received_req->WalkLen(), 3);
    int32_t cursor = 0;
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(*(received_req->GetSrcIds() + i), src_ids[i]);
      EXPECT_EQ(*(received_req->GetParentIds() + i), parent_ids[i]);
      for (int32_t j = 0; j < i; ++j) {
        EXPECT_EQ(*(received_req->GetParentNeighborIds() + cursor + j), parent_ids[i]);
      }
      cursor += *(received_req->GetParentNeighborSegments() + i);
    }
    delete received_req;
  }

  {
    // Fill response for serialize
    RandomWalkResponse res;
    res.InitWalks(4);
    res.SetBatchSize(4);

    int64_t walks[4];
    for (int32_t i = 0; i < 4; ++i) {
      walks[i] = i + 1;
    }

    res.AppendWalks(walks, 4);

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.batch_size_, 4);
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(res.tensors_[kNodeIds].GetInt64(i), i + 1);
      EXPECT_EQ(*(res.GetWalks() + i), i + 1);
    }

    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    RandomWalkResponse* received_res = new RandomWalkResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->batch_size_, 4);
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(received_res->tensors_[kNodeIds].GetInt64(i), i + 1);
    }
    delete received_res;
  }

  {
    // Fill response for serialize
    RandomWalkResponse res;

    res.InitWalks(4);
    res.SetBatchSize(4);
    res.InitNeighbors(4, 10);

    int64_t walks[4];
    int32_t degrees[4];
    for (int32_t i = 0; i < 4; ++i) {
      walks[i] = i + 1;
      degrees[i] = i + 1;
    }

    int64_t nbrs[10];
    int32_t cursor = 0;
    for (int32_t i = 0; i < 4; ++i) {
      for (int32_t j = 1; j < i + 1; ++j) {
        nbrs[cursor] = j;
        ++cursor;
      }
    }

    res.AppendWalks(walks, 4);
    res.AppendNeighborIds(nbrs, 10);
    res.AppendDegrees(degrees, 4);

    // Parse response after deserialize
    // In memory
    EXPECT_EQ(res.batch_size_, 4);
    int32_t idx = 0;
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(res.tensors_[kNodeIds].GetInt64(i), i + 1);
      EXPECT_EQ(*(res.GetWalks() + i), i + 1);
      EXPECT_EQ(*(res.GetDegrees() + i), i + 1);
      for (int32_t j = 1; j < i + 1; ++j) {
        EXPECT_EQ(*(res.GetNeighborIds() + idx), j);
        ++idx;
      }
    }

    // RPC
    OpResponsePb* pb_res = new OpResponsePb();
    RandomWalkResponse* received_res = new RandomWalkResponse();
    res.SerializeTo(pb_res);
    received_res->ParseFrom(pb_res);
    EXPECT_EQ(received_res->batch_size_, 4);
    idx = 0;
    for (int32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(received_res->tensors_[kNodeIds].GetInt64(i), i + 1);
      EXPECT_EQ(*(received_res->GetWalks() + i), i + 1);
      EXPECT_EQ(*(received_res->GetDegrees() + i), i + 1);
      for (int32_t j = 1; j < i + 1; ++j) {
        EXPECT_EQ(*(received_res->GetNeighborIds() + idx), j);
        ++idx;
      }
    }
    delete received_res;
  }
}
