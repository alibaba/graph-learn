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

#include <algorithm>

#include "brane/core/shard-config.hh"
#include "actor/graph/sharded_graph_store.h"
#include "actor/test/test_env.h"
#include "actor/utils.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

class ActorTest : public ::testing::Test {
  using NeighborsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
public:
  ActorTest() = default;
  ~ActorTest() override = default;

protected:
  void SetUp() override {
    env_.Initialize();
  }

  void TearDown() override {
    env_.Finalize();
  }

  NeighborsMap LoadEdgeData(const char* fname) {
    NeighborsMap neighbors;
    std::ifstream in(fname);
    std::stringstream ss;
    std::string line, ip;
    int64_t src_id = 0, dst_id = 0;
    float edge_weight = 0.0;
    // Ignore the first line.
    std::getline(in, line);
    while (in >> src_id >> dst_id >> edge_weight) {
      if (neighbors.find(src_id) == neighbors.end()) {
        neighbors[src_id] = std::vector<int64_t>{dst_id};
      } else {
        neighbors[src_id].push_back(dst_id);
      }
    }
    return neighbors;
  }

private:
  actor::TestEnv env_;
};

TEST_F(ActorTest, NodeCorrectness) {
  auto &sgs = actor::ShardedGraphStore::Get();
  auto num_shards = brane::local_shard_count();
  for (uint32_t i = 0; i < num_shards; ++i) {
    for (auto type : {"user", "item"}) {
      auto noder = sgs.OnShard(i)->GetNoder(type);
      EXPECT_TRUE(noder != nullptr);
      auto data_size = noder->GetLocalStorage()->Size();
      EXPECT_EQ(40 + 5 * (i + 1), data_size);
      auto ids = *(noder->GetLocalStorage()->GetIds());
      std::sort(ids.begin(), ids.end());
      uint32_t counter = i;
      for (auto &id : ids) {
        EXPECT_EQ(counter, id);
        counter += num_shards;
      }
    }
  }
}

TEST_F(ActorTest, EdgeCorrectness) {
  auto &sgs = actor::ShardedGraphStore::Get();
  auto num_shards = brane::local_shard_count();
  for (uint32_t i = 0; i < num_shards; ++i) {
    for (auto pair : {
          std::make_pair("click", "user_to_item_weighted_edge_file"),
          std::make_pair("similar", "item_to_item_weighted_edge_file")
        }) {
      auto graph = sgs.OnShard(i)->GetGraph(pair.first)->GetLocalStorage();
      auto neibors_map = LoadEdgeData(pair.second);
      auto src_ids = graph->GetAllSrcIds();
      for (auto &src_id : *src_ids) {
        auto tmp = graph->GetNeighbors(src_id);
        std::vector<int64_t> neighbors;
        neighbors.reserve(tmp.Size());
        for (int32_t nid = 0; nid < tmp.Size(); ++nid) {
          neighbors.push_back(tmp[nid]);
        }
        auto expect_neighbors = neibors_map.find(src_id)->second;
        std::sort(neighbors.begin(), neighbors.end());
        std::sort(expect_neighbors.begin(), expect_neighbors.end());
        EXPECT_EQ(expect_neighbors, neighbors);
      }
    }
  }
}
