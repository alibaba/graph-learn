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
#include "include/constants.h"
#include "include/sampling_request.h"
#include "include/random_walk_request.h"
#include "core/partition/partitioner.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class PartitionStitchTest : public ::testing::Test {
public:
  PartitionStitchTest() {
    InitGoogleLogging();
  }
  ~PartitionStitchTest() {
    UninitGoogleLogging();
  }
};

TEST_F(PartitionStitchTest, DenseReq_SparseRes) {
  SamplingRequest req("i-i", "FullSampler", 3);
  SamplingResponse* res = new SamplingResponse;
  int64_t src_ids[4] = {0, 1, 2, 3};
  req.Set(src_ids, 4);

  static PartitionerCreator<OpRequest> creator(2);
  auto partitioner = creator(PartitionMode::kByHash);
  ShardsPtr<OpRequest> req_shards = partitioner->Partition(&req);

  SamplingResponse* res1 = new SamplingResponse;
  std::vector<int32_t> inds{0, 2};
  res1->SetShape(2, 6, inds);
  res1->InitNeighborIds();
  res1->InitEdgeIds();

  // 0->, 2->2, 2->3
  for (auto i : inds) {
    for (int32_t j = 0; j < i; ++j) {
      res1->AppendNeighborId(i + j);
      res1->AppendEdgeId(i + j);
    }
  }

  SamplingResponse* res2 = new SamplingResponse;
  inds = {1, 3};
  res2->SetShape(2, 6, inds);
  res2->InitNeighborIds();
  res2->InitEdgeIds();

  // 1->1, 3->3, 3->4, 3->5
  for (auto i : inds) {
    for (int32_t j = 0; j < i; ++j) {
      res2->AppendNeighborId(i + j);
      res2->AppendEdgeId(i + j);
    }
  }

  ShardsPtr<OpResponse> res_shards(
        new Shards<OpResponse>(req_shards->Capacity()));

  res_shards->Add(0, res1, true);
  res_shards->Add(1, res2, true);

  res_shards->StickerPtr()->CopyFrom(*(req_shards->StickerPtr()));
  res->Stitch(res_shards);

  EXPECT_EQ(res->GetShape().dim1, 4);
  EXPECT_EQ(res->GetShape().dim2, 6);
  EXPECT_EQ(res->GetShape().size, 6);
  std::vector<int32_t> segments{0, 1, 2, 3};
  int32_t cursor = 0;
  for (int32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(res->GetShape().segments[i], segments[i]);
    for (int32_t j = i; j < 2 * i; ++j) {
      EXPECT_EQ(*(res->GetNeighborIds() + cursor), j);
      EXPECT_EQ(*(res->GetEdgeIds() + cursor), j);
      cursor++;
    }
  }
}

TEST_F(PartitionStitchTest, DenseReq_DenseRes) {
  SamplingRequest req("i-i", "RandomSampler", 3);
  SamplingResponse* res = new SamplingResponse;
  int64_t src_ids[4] = {1, 2, 3, 4};
  req.Set(src_ids, 4);

  static PartitionerCreator<OpRequest> creator(2);
  auto partitioner = creator(PartitionMode::kByHash);
  ShardsPtr<OpRequest> req_shards = partitioner->Partition(&req);

  SamplingResponse* res1 = new SamplingResponse;
  res1->SetShape(2, 6);
  res1->InitNeighborIds();
  res1->InitEdgeIds();

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 6; ++j) {
      res1->AppendNeighborId(j);
      res1->AppendEdgeId(j);
    }
  }

  SamplingResponse* res2 = new SamplingResponse;
  res2->SetShape(2, 6);
  res2->InitNeighborIds();
  res2->InitEdgeIds();

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 6; ++j) {
      res2->AppendNeighborId(j);
      res2->AppendEdgeId(j);
    }
  }

  ShardsPtr<OpResponse> res_shards(
        new Shards<OpResponse>(req_shards->Capacity()));

  res_shards->Add(0, res1, true);
  res_shards->Add(1, res2, true);

  res_shards->StickerPtr()->CopyFrom(*(req_shards->StickerPtr()));
  res->Stitch(res_shards);

  EXPECT_EQ(res->GetShape().dim1, 4);
  EXPECT_EQ(res->GetShape().dim2, 6);
  EXPECT_EQ(res->GetShape().size, 24);

  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 6; ++j) {
      EXPECT_EQ(*(res->GetNeighborIds() + i * 6 + j), j);
      EXPECT_EQ(*(res->GetEdgeIds() + i * 6 + j), j);
    }
  }
}

TEST_F(PartitionStitchTest, SparseReq_SparseRes) {
  // Edges:
  // 0->, 1->1, 2->2, 2->3, 3->3, 3->4, 3->5

  RandomWalkRequest req("i-i", 0.25, 0.25, 3);
  RandomWalkResponse* res = new RandomWalkResponse;
  int64_t src_ids[4] = {0, 1, 2, 3};
  int64_t parent_ids[4] = {0, 1, 2, 3};
  int32_t batch_size = 4;
  int64_t parent_neighbor_ids[6] = {1, 2, 3, 3, 4, 5};
  int32_t parent_neighbor_segments[4] = {0, 1, 2, 3};
  int32_t total_count = 6;

  req.Set(src_ids, parent_ids, 4, parent_neighbor_ids, parent_neighbor_segments, total_count);

  static PartitionerCreator<OpRequest> creator(2);
  auto partitioner = creator(PartitionMode::kByHash);
  ShardsPtr<OpRequest> req_shards = partitioner->Partition(&req);

  int32_t shard_id = 0;
  OpRequest* shard_req = nullptr;
  req_shards->Next(&shard_id, &shard_req);
  const RandomWalkRequest* request1 =
    static_cast<const RandomWalkRequest*>(shard_req);
  EXPECT_EQ(shard_id, 0);
  EXPECT_EQ(request1->IsDeepWalk(), false);
  EXPECT_EQ(request1->WalkLen(), 3);
  EXPECT_EQ(request1->BatchSize(), 2);
  int32_t start = 0;
  for (int32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(*(request1->GetSrcIds() + i), 2 * i);
    EXPECT_EQ(*(request1->GetParentIds() + i), 2 * i);
    for (int32_t j = 2 * i; j < 4 * i; ++j) {
      EXPECT_EQ(*(request1->GetParentNeighborIds() + start), j);
      ++start;
    }
    EXPECT_EQ(*(request1->GetParentNeighborSegments() + i), 2 * i);
  }

  req_shards->Next(&shard_id, &shard_req);
  const RandomWalkRequest* request2 =
    static_cast<const RandomWalkRequest*>(shard_req);
  EXPECT_EQ(shard_id, 1);
  EXPECT_EQ(request2->IsDeepWalk(), false);
  EXPECT_EQ(request2->WalkLen(), 3);
  EXPECT_EQ(request2->BatchSize(), 2);
  start = 0;
  for (int32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(*(request2->GetSrcIds() + i), 2 * i + 1);
    EXPECT_EQ(*(request2->GetParentIds() + i), 2 * i + 1);
    for (int32_t j = 2 * i + 1; j < 4 * i + 2; ++j) {
      EXPECT_EQ(*(request2->GetParentNeighborIds() + start), j);
      ++start;
    }
    EXPECT_EQ(*(request2->GetParentNeighborSegments() + i), 2 * i + 1);
  }

  // 0 -> -1 -> -1 -> -1
  // 2 -> 3 -> 4 -> 5
  RandomWalkResponse* res1 = new RandomWalkResponse;
  res1->InitWalks(3);
  res1->InitNeighbors(2, 2);
  res1->SetBatchSize(2);
  int64_t walks1[6] = {-1, -1, -1, 3, 4, 5};
  int64_t neighbor_ids1[2] = {2, 3};
  int32_t degrees1[2] = {0, 2};
  res1->AppendWalks(walks1, 6);
  res1->AppendNeighborIds(neighbor_ids1, 2);
  res1->AppendDegrees(degrees1, 2);

  // 1 -> 1 -> 1 -> 1
  // 3 -> 4 -> 5 -> 6
  RandomWalkResponse* res2 = new RandomWalkResponse;
  res2->InitWalks(3);
  res2->InitNeighbors(2, 4);
  res2->SetBatchSize(2);
  int64_t walks2[6] = {1, 1, 1, 4, 5, 6};
  int64_t neighbor_ids2[4] = {1, 3, 4, 5};
  int32_t degrees2[2] = {1, 3};
  res2->AppendWalks(walks2, 6);
  res2->AppendNeighborIds(neighbor_ids2, 4);
  res2->AppendDegrees(degrees2, 2);

  ShardsPtr<OpResponse> res_shards(
        new Shards<OpResponse>(req_shards->Capacity()));

  res_shards->Add(0, res1, true);
  res_shards->Add(1, res2, true);

  res_shards->StickerPtr()->CopyFrom(*(req_shards->StickerPtr()));
  res->Stitch(res_shards);

  for (int32_t j = 0; j < 3; ++j) {
    EXPECT_EQ(*(res->GetWalks() + j), -1);
    EXPECT_EQ(*(res->GetWalks() + 3 + j), 1);
  }
  for (int32_t i = 2; i < 4; ++i) {
    for (int32_t j = 0; j < 3; ++j) {
      EXPECT_EQ(*(res->GetWalks() + i * 3 + j), i + j + 1);
    }
  }
  int32_t cursor = 0;
  for (int32_t i = 0; i < 4; ++i) {
    for (int32_t j = i; j < 2 * i; ++j) {
      EXPECT_EQ(*(res->GetNeighborIds() + cursor), j);
      cursor++;
    }
    EXPECT_EQ(*(res->GetDegrees() + i), i);
  }
}