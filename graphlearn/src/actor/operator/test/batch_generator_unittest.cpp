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

#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "seastar/core/alien.hh"
#include "seastar/core/when_all.hh"

#include "actor/test/test_env.h"
#include "actor/dag/dag_actor_manager.h"
#include "actor/operator/batch_generator.h"
#include "actor/service/actor_alien.h"
#include "platform/protobuf.h"

#include "actor/generated/operator/graph_ops_ref.act.autogen.h"
#include "generated/proto/dag.pb.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::act;  // NOLINT [build/namespaces]

const char* template_dag_content =
  "nodes {{ \n"
    "id: 1 \n"
    "op_name: \"GetNodes\" \n"
    "params {{ \n"
      "name: \"nf\" \n"
      "length: 1 \n"
      "int32_values: 2 \n"
    "}} \n"
    "params {{ \n"
      "name: \"nt\" \n"
      "dtype: 4 \n"
      "length: 1 \n"
      "string_values: \"user\" \n"
    "}} \n"
    "params {{ \n"
      "name: \"ep\" \n"
      "length: 1 \n"
      "int32_values: 2147483647 \n"
    "}} \n"
    "params {{ \n"
      "name: \"bs\" \n"
      "length: 1 \n"
      "int32_values: 8 \n"
    "}} \n"
    "params {{ \n"
      "name: \"str\" \n"
      "dtype: 4 \n"
      "length: 1 \n"
      "string_values: \"{0}\" \n"
    "}} \n"
    "out_edges {{ \n"
      "id: 1 \n"
      "src_output: \"nid\" \n"
      "dst_input: \"ids\" \n"
    "}} \n"
  "}} \n"
  "nodes {{ \n"
    "id: 2 \n"
    "op_name: \"GetEdges\" \n"
    "params {{ \n"
      "name: \"et\" \n"
      "dtype: 4 \n"
      "length: 1 \n"
      "string_values: \"click\" \n"
    "}} \n"
    "params {{ \n"
      "name: \"ep\" \n"
      "length: 1 \n"
      "int32_values: 2147483647 \n"
    "}} \n"
    "params {{ \n"
      "name: \"bs\" \n"
      "length: 1 \n"
      "int32_values: 8 \n"
    "}} \n"
    "params {{ \n"
      "name: \"str\" \n"
      "dtype: 4 \n"
      "length: 1 \n"
      "string_values: \"{0}\" \n"
    "}} \n"
  "}} \n"
  "nodes {{ \n"
    "id: 3 \n "
    "op_name: \"Sink\" \n "
    "in_edges {{ \n "
      "id: 1 \n"
      "src_output: \"nid\" \n"
      "dst_input: \"ids\" \n"
    "}} \n"
  "}}";

const unsigned node_getter_dag_id = 1;
const unsigned edge_getter_dag_id = 2;

struct edge_record {
  io::IdType edge_id;
  io::IdType src_id;
  io::IdType dst_id;

  edge_record(io::IdType edge_id, io::IdType src_id, io::IdType dst_id)
    : edge_id(edge_id), src_id(src_id), dst_id(dst_id) {}
  edge_record(const edge_record&) = default;
  edge_record(edge_record&&) = default;
  ~edge_record() = default;

  edge_record& operator=(const edge_record&) = default;
  edge_record& operator=(edge_record&&) = default;
};
inline bool operator==(const edge_record& lhs, const edge_record& rhs) {
  return ((lhs.edge_id == rhs.edge_id) && (lhs.src_id == rhs.src_id)
    && (lhs.dst_id == rhs.dst_id));
}
inline bool operator<(const edge_record& lhs, const edge_record& rhs) {
  if (lhs.src_id != rhs.src_id) {
    return lhs.src_id < rhs.src_id;
  } else if (lhs.edge_id != rhs.edge_id) {
    return lhs.edge_id < rhs.edge_id;
  } else {
    return lhs.dst_id < rhs.dst_id;
  }
}

class BatchGeneratorUnitTest : public ::testing::Test {
public:
  BatchGeneratorUnitTest() = default;
  ~BatchGeneratorUnitTest() override = default;

protected:
  void SetUp() override {}

  void TearDown() override {}

  void NodeTestImplFunc(TestEnv* env, const std::string& strategy) {
    // Init
    env->Initialize();
    RegisterDag(strategy);

    // Prepare checking datas
    ShardDataInfoVecT infos;
    auto data_size = GetSortedInfosOfNodes(&infos);
    auto intact_batch_num = GetIntactBatchNum(infos);
    auto batch_to_shard = GetBatchIdToShardMap(infos, data_size);
    auto expect = GetExpectNodeDataArray(
        infos, batch_to_shard, intact_batch_num, data_size);
    EXPECT_EQ(expect.size(), data_size);

    // Run 3 epoches and test
    auto refs = CreateNodeGetterRefs();
    for (uint32_t i = 0; i < 3; i++) {
      auto fut = seastar::alien::submit_to(
          *default_alien,
          0,
          [&strategy, data_size, &batch_to_shard, &expect, &refs] () mutable {
        return FetchAllNodeBatch(refs, batch_to_shard, data_size).then(
            [&strategy, data_size, &expect] (std::vector<io::IdType> results) {
          EXPECT_EQ(results.size(), data_size);
          if (strategy == "shuffled") {
            std::sort(expect.begin(), expect.end(), std::less<io::IdType>{});
            std::sort(results.begin(), results.end(), std::less<io::IdType>{});
          }
          for (uint32_t j = 0; j < data_size; j++) {
            EXPECT_TRUE(expect[j] == results[j]);
          }
        });
      });
      fut.wait();
    }
    for (auto& op_ref : refs) {
      delete op_ref;
    }

    env->Finalize();
  }

  void EdgeTestImplFunc(TestEnv* env, const std::string& strategy) {
    // Init
    env->Initialize();
    RegisterDag(strategy);

    // Prepare checking datas
    ShardDataInfoVecT infos;
    auto data_size = GetSortedInfosOfEdges(&infos);
    auto intact_batch_num = GetIntactBatchNum(infos);
    auto batch_to_shard = GetBatchIdToShardMap(infos, data_size);
    auto expect = GetExpectEdgeDataArray(infos, batch_to_shard,
      intact_batch_num, data_size);
    EXPECT_EQ(expect.size(), data_size);

    // Run 3 epoches and test
    auto refs = CreateEdgeGetterRefs();
    for (uint32_t i = 0; i < 3; i++) {
      auto fut = seastar::alien::submit_to(
          *default_alien,
          0,
          [&strategy, data_size, &batch_to_shard, &expect, &refs] () mutable {
        return FetchAllEdgeBatch(refs, batch_to_shard, data_size).then(
            [&strategy, data_size, &expect] (std::vector<edge_record> results) {
          EXPECT_EQ(results.size(), data_size);
          if (strategy == "shuffled") {
            std::sort(expect.begin(), expect.end(), std::less<edge_record>{});
            std::sort(results.begin(), results.end(), std::less<edge_record>{});
          }
          for (uint32_t j = 0; j < data_size; j++) {
            EXPECT_TRUE(expect[j] == results[j]);
          }
        });
      });
      fut.wait();
    }
    for (auto& op_ref : refs) {
      delete op_ref;
    }

    env->Finalize();
  }

private:
  void RegisterDag(const std::string& strategy) {
    std::string dag_content =
      fmt::format(template_dag_content, strategy);
    DagDef def;
    PB_NAMESPACE::TextFormat::ParseFromString(dag_content, &def);
    Dag* dag = new Dag(def);
    dag_id_ = dag->Id();
    auto& dag_manager = DagActorManager::GetInstance();
    dag_manager.Clear();
    dag_manager.AddDag(dag);
    delete dag;
  }

  std::vector<BaseOperatorActor_ref*> CreateNodeGetterRefs() const {
    std::vector<BaseOperatorActor_ref*> refs;
    refs.reserve(hiactor::local_shard_count());
    for (uint32_t i = 0; i < hiactor::local_shard_count(); ++i) {
      hiactor::scope_builder builder(i + hiactor::machine_info::sid_anchor());
      refs.push_back(builder.new_ref<NodeGetterActor_ref>(
          MakeActorGUID(dag_id_, node_getter_dag_id)));
    }
    return refs;
  }

  std::vector<BaseOperatorActor_ref*> CreateEdgeGetterRefs() const {
    std::vector<BaseOperatorActor_ref*> refs;
    refs.reserve(hiactor::local_shard_count());
    for (uint32_t i = 0; i < hiactor::local_shard_count(); ++i) {
      hiactor::scope_builder builder(i + hiactor::machine_info::sid_anchor());
      refs.push_back(builder.new_ref<EdgeGetterActor_ref>(
          MakeActorGUID(dag_id_, edge_getter_dag_id)));
    }
    return refs;
  }

  static int64_t GetSortedInfosOfNodes(ShardDataInfoVecT* data_info_vec) {
    data_info_vec->reserve(hiactor::local_shard_count());
    int64_t total_size = 0;
    for (int32_t i = 0; i < hiactor::local_shard_count(); ++i) {
      auto noder = ShardedGraphStore::Get().OnShard(i)->GetNoder("user");
      auto shard_data_size = noder->GetLocalStorage()->Size();
      total_size += shard_data_size;
      data_info_vec->emplace_back(shard_data_size, i);
    }
    std::sort(data_info_vec->begin(), data_info_vec->end(), DataSizeLess);
    return total_size;
  }

  static int64_t GetSortedInfosOfEdges(ShardDataInfoVecT* data_info_vec) {
    data_info_vec->reserve(hiactor::local_shard_count());
    int64_t total_size = 0;
    for (int32_t i = 0; i < hiactor::local_shard_count(); ++i) {
      auto graph = ShardedGraphStore::Get().OnShard(i)->GetGraph("click");
      auto shard_data_size = graph->GetLocalStorage()->GetEdgeCount();
      total_size += shard_data_size;
      data_info_vec->emplace_back(shard_data_size, i);
    }
    std::sort(data_info_vec->begin(), data_info_vec->end(), DataSizeLess);
    return total_size;
  }

  std::vector<uint32_t>
  GetBatchIdToShardMap(const ShardDataInfoVecT& sorted_infos,
                       int64_t total_data_size) const {
    auto total_batches = total_data_size / batch_size_ +
                         (total_data_size % batch_size_ ? 1 : 0);
    std::vector<uint32_t> batch_to_shard;
    batch_to_shard.resize(total_batches);

    uint32_t batch_id = 0;
    uint32_t offset = 0;
    auto local_shards = hiactor::local_shard_count();
    for (uint32_t i = 0; i < local_shards; i++) {
      auto shard_total_batch_num = sorted_infos[i].data_size / batch_size_;
      for (auto j = offset; j < shard_total_batch_num; j++) {
        for (uint32_t behind_i = i; behind_i < local_shards; behind_i++) {
          batch_to_shard[batch_id++] = sorted_infos[behind_i].shard_id;
        }
      }
      offset = shard_total_batch_num;
    }
    for (uint32_t i = 0; batch_id < total_batches; batch_id++) {
      batch_to_shard[batch_id] = sorted_infos[i++].shard_id;
    }
    return batch_to_shard;
  }

  unsigned GetIntactBatchNum(const ShardDataInfoVecT& sorted_infos) const {
    unsigned intact_batch_num = 0;
    for (auto& sorted_info : sorted_infos) {
      intact_batch_num += sorted_info.data_size / batch_size_;
    }
    return intact_batch_num;
  }

  std::vector<io::IdType>
  GetExpectNodeDataArray(const ShardDataInfoVecT& sorted_infos,
                         const std::vector<uint32_t>& batch_to_shard,
                         unsigned intact_batch_num,
                         int64_t total_data_size) const {
    auto local_shards = hiactor::local_shard_count();
    std::vector<const io::IdType*> id_arrays;
    for (int32_t i = 0; i < local_shards; i++) {
      auto noder = ShardedGraphStore::Get().OnShard(i)->GetNoder("user");
      id_arrays.push_back(noder->GetLocalStorage()->GetIds().data());
    }
    std::vector<unsigned> offsets;
    offsets.resize(local_shards, 0);

    std::vector<io::IdType> expect_datas;
    expect_datas.reserve(total_data_size);
    for (uint32_t i = 0; i < intact_batch_num; i++) {
      auto shard = batch_to_shard[i];
      for (uint32_t j = 0; j < batch_size_; j++) {
        expect_datas.push_back(id_arrays[shard][offsets[shard]]);
        offsets[shard]++;
      }
    }
    for (uint32_t i = 0; i < sorted_infos.size(); i++) {
      auto shard = sorted_infos[i].shard_id;
      while (offsets[shard] < sorted_infos[shard].data_size) {
        expect_datas.push_back(id_arrays[shard][offsets[shard]]);
        offsets[shard]++;
      }
    }
    return expect_datas;
  }

  std::vector<edge_record>
  GetExpectEdgeDataArray(const ShardDataInfoVecT& sorted_infos,
                         const std::vector<uint32_t>& batch_to_shard,
                         unsigned intact_batch_num,
                         int64_t total_data_size) const {
    auto local_shards = hiactor::local_shard_count();
    std::vector<io::GraphStorage*> edge_stores;
    for (int32_t i = 0; i < local_shards; i++) {
      auto graph = ShardedGraphStore::Get().OnShard(i)->GetGraph("click");
      edge_stores.push_back(graph->GetLocalStorage());
    }
    std::vector<unsigned> offsets;
    offsets.resize(local_shards, 0);

    std::vector<edge_record> expect_datas;
    expect_datas.reserve(total_data_size);
    for (uint32_t i = 0; i < intact_batch_num; i++) {
      auto shard = batch_to_shard[i];
      for (uint32_t j = 0; j < batch_size_; j++) {
        expect_datas.emplace_back(offsets[shard],
          edge_stores[shard]->GetSrcId(offsets[shard]),
          edge_stores[shard]->GetDstId(offsets[shard]));
        offsets[shard]++;
      }
    }
    for (uint32_t i = 0; i < sorted_infos.size(); i++) {
      auto shard = sorted_infos[i].shard_id;
      while (offsets[shard] < sorted_infos[shard].data_size) {
        expect_datas.emplace_back(offsets[shard],
          edge_stores[shard]->GetSrcId(offsets[shard]),
          edge_stores[shard]->GetDstId(offsets[shard]));
        offsets[shard]++;
      }
    }
    return expect_datas;
  }

  static seastar::future<std::vector<io::IdType>>
  FetchAllNodeBatch(const std::vector<BaseOperatorActor_ref*>& op_refs,
                    const std::vector<uint32_t>& batch_to_shard,
                    int64_t total_data_size) {
    std::vector<seastar::future<TensorMapSerializer>> futs;
    futs.reserve(batch_to_shard.size());
    for (auto dest_shard : batch_to_shard) {
      futs.emplace_back(op_refs[dest_shard]->Process(TensorMapSerializer{}));
    }
    return seastar::when_all(futs.begin(), futs.end()).then(
        [total_data_size] (std::vector<seastar::future<TensorMapSerializer>> results) {
      std::vector<io::IdType> data;
      data.reserve(total_data_size);
      for (auto& result : results) {
        auto res_tm = result.get0();
        for (auto &tn : res_tm.tensors_) {
          auto size = tn.second.Size();
          const int64_t* begin = tn.second.GetInt64();
          data.insert(data.end(), begin, begin + size);
        }
      }
      return seastar::make_ready_future<std::vector<io::IdType>>(
          std::move(data));
    });
  }

  static seastar::future<std::vector<edge_record>>
  FetchAllEdgeBatch(const std::vector<BaseOperatorActor_ref*>& op_refs,
                    const std::vector<uint32_t>& batch_to_shard,
                    int64_t total_data_size) {
    std::vector<seastar::future<TensorMapSerializer>> futs;
    futs.reserve(batch_to_shard.size());
    for (auto dest_shard : batch_to_shard) {
      futs.emplace_back(op_refs[dest_shard]->Process(TensorMapSerializer{}));
    }
    return seastar::when_all(futs.begin(), futs.end()).then(
        [total_data_size] (std::vector<seastar::future<TensorMapSerializer>> results) {
      std::vector<edge_record> data;
      data.reserve(total_data_size);
      for (auto& result : results) {
        auto res_tm = result.get0();
        auto edge_len = res_tm.tensors_[kEdgeIds].Size();
        auto* edge_ptr = res_tm.tensors_[kEdgeIds].GetInt64();
        auto* src_ptr = res_tm.tensors_[kSrcIds].GetInt64();
        auto* dst_ptr = res_tm.tensors_[kDstIds].GetInt64();
        for (int j = 0; j < edge_len; ++j) {
          data.emplace_back(edge_ptr[j], src_ptr[j], dst_ptr[j]);
        }
      }
      return seastar::make_ready_future<std::vector<edge_record>>(
          std::move(data));
    });
  }

private:
  int32_t          dag_id_ = 0;
  const unsigned   batch_size_ = 8;
};

// Node batch generator unit tests
TEST_F(BatchGeneratorUnitTest, Node_Ordered_SkewedWithPieceBatch) {
  TestEnv env(10240, 2, 40, 5, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Node_Ordered_SkewedWithoutPieceBatch) {
  TestEnv env(10240, 2, 40, 8, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Node_Ordered_BalancedWithPieceBatch) {
  TestEnv env(10240, 2, 45, 0, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Node_Ordered_BalancedWithoutPieceBatch) {
  TestEnv env(10240, 2, 48, 0, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Node_Ordered_BatchCrossShard) {
  TestEnv env(10240, 2, 2, 2, 10, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Node_Shuffled_SkewedWithPieceBatch) {
  TestEnv env(10240, 2, 40, 5, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Node_Shuffled_SkewedWithoutPieceBatch) {
  TestEnv env(10240, 2, 40, 8, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Node_Shuffled_BalancedWithPieceBatch) {
  TestEnv env(10240, 2, 45, 0, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Node_Shuffled_BalancedWithoutPieceBatch) {
  TestEnv env(10240, 2, 48, 0, 100, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Node_Shuffled_BatchCrossShard) {
  TestEnv env(10240, 2, 2, 2, 10, 1, 0, TestEnv::DatasetCode::kValid);
  NodeTestImplFunc(&env, "shuffled");
}

// Edge batch generator unit tests
TEST_F(BatchGeneratorUnitTest, Edge_Ordered_WithoutPieceBatch) {
  TestEnv env(10240, 2, 100, 0, 1000, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Edge_Ordered_WithPieceBatch) {
  TestEnv env(10240, 2, 105, 0, 1000, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Edge_Ordered_BatchCrossShard) {
  TestEnv env(10240, 2, 3, 0, 12, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "by_order");
}

TEST_F(BatchGeneratorUnitTest, Edge_Shuffled_WithoutPieceBatch) {
  TestEnv env(10240, 2, 100, 0, 1000, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Edge_Shuffled_WithPieceBatch) {
  TestEnv env(10240, 2, 105, 0, 1000, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "shuffled");
}

TEST_F(BatchGeneratorUnitTest, Edge_Shuffled_BatchCrossShard) {
  TestEnv env(10240, 2, 3, 0, 12, 1, 0, TestEnv::DatasetCode::kValid);
  EdgeTestImplFunc(&env, "shuffled");
}
