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

#include <memory>

#include "gtest/gtest.h"

#include "hiactor/core/actor-app.hh"

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "core/execution/dag.h"
#include "core/execution/query_executor.h"
#include "generated/fbs/query_plan_generated.h"

using namespace dgs;

void MockSampleStore(std::unique_ptr<storage::SampleStore>& store) {
  VertexType            vtype_ = 0;
  EdgeType              etype_ = 4;
  VertexId              vid_ = 2;
  OperatorId            vop_ = 2;
  OperatorId            eop1_ = 1;
  OperatorId            eop2_ = 3;

  ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");

  auto partitioner = PartitionerFactory::Create("hash", 1);
  std::vector<PartitionId> pids{0};
  store = std::make_unique<storage::SampleStore>(
        pids, std::move(partitioner),
        "./tmp_store/", "./tmp_store/",
        storage::RdbEnv::Default());

  /// Nodes for opid=2
  /// id attributes
  /// 2  "2"
  /// 2  "attr2"
  /// 3  "3"
  /// 3  "attr2"
  /// 4  "4"
  /// 4  "attr2"
  /// 5  "5"
  /// 5  "attr2"
  /// 6  "6"
  /// 6  "attr2"

  /// Edges for opid=1, which is 1hop sample
  /// src_id dst_id
  /// 2      3
  /// 2      4

  /// Edges for opid=3, which is 2hop sample
  /// src_id dst_id
  /// 3      4
  /// 3      5
  /// 4      5
  /// 4      6

  /// Query Run with input vid=2, 1hop sample(op_id=1) returns 2->3, 2->4;
  /// 2hop sample(op_id=3) returns 3->4, 3->5, 4->5, 4->6;
  /// Lookup node features(op_id=2) with 2 versions for each vid in [2, 3, 4, 5, 6].

  // Put vertices.
  [vtype_, vid_, vop_, store = store.get()] {
    io::RecordBuilder builder;
    for (VertexId vid = vid_; vid < vid_ + 6; ++vid) {
      storage::Key vkey1{vtype_, vid, vop_, 0};
      storage::Key vkey2{vtype_, vid, vop_, 1};

      builder.Clear();
      std::string attr1 = std::to_string(vid);
      builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
      builder.BuildAsVertexRecord(vtype_, vid);
      store->PutVertex(vkey1, {builder.BufPointer(), builder.BufSize()});

      builder.Clear();
      attr1 = "attr2";
      builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
      builder.BuildAsVertexRecord(vtype_, vid);
      store->PutVertex(vkey2, {builder.BufPointer(), builder.BufSize()});
      std::vector<storage::KVPair> ret;
      store->GetVerticesByPrefix(vkey1.pkey, &ret);
    }
  }();

  // Put edges for eop1
  [vtype_, vid_, eop1_, etype_, store = store.get()] {
    storage::Key ekey1{vtype_, vid_, eop1_, 0};
    storage::Key ekey2{vtype_, vid_, eop1_, 1};

    io::RecordBuilder builder;
    std::string attr1 = "attr1";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_, vid_ + 1);
    store->PutEdge(ekey1, {builder.BufPointer(), builder.BufSize()});

    builder.Clear();
    attr1 = "attr2";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_, vid_ + 2);
    store->PutEdge(ekey2, {builder.BufPointer(), builder.BufSize()});

    dgs::io::Record record;
    store->GetEdge(ekey1, &record);
    store->GetEdge(ekey2, &record);

    std::vector<storage::KVPair> ret;
    store->GetEdgesByPrefix(ekey1.pkey, &ret);
  }();

  // Put edges for eop2
  [vtype_, vid_, eop2_, etype_, store = store.get()] {
    storage::Key ekey1{vtype_, vid_ + 1, eop2_, 0};
    storage::Key ekey2{vtype_, vid_ + 1, eop2_, 1};
    storage::Key ekey3{vtype_, vid_ + 2, eop2_, 0};
    storage::Key ekey4{vtype_, vid_ + 2, eop2_, 1};

    io::RecordBuilder builder;
    std::string attr1 = "attr1";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_ + 1, vid_ + 2);
    store->PutEdge(ekey1, {builder.BufPointer(), builder.BufSize()});

    builder.Clear();
    attr1 = "attr2";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_ + 1, vid_ + 3);
    store->PutEdge(ekey2, {builder.BufPointer(), builder.BufSize()});

    builder.Clear();
    attr1 = "attr3";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_ + 2, vid_ + 3);
    store->PutEdge(ekey3, {builder.BufPointer(), builder.BufSize()});

    builder.Clear();
    attr1 = "attr4";
    builder.AddAttribute(0, AttributeValueType::STRING, reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid_ + 2, vid_ + 4);
    store->PutEdge(ekey4, {builder.BufPointer(), builder.BufSize()});

    dgs::io::Record record;
    store->GetEdge(ekey1, &record);
    store->GetEdge(ekey2, &record);
    store->GetEdge(ekey3, &record);
    store->GetEdge(ekey4, &record);

    std::vector<storage::KVPair> ret;
    store->GetEdgesByPrefix(ekey1.pkey, &ret);
  }();
}

class QueryExecutorTester {
public:
  QueryExecutorTester() {
    MockSampleStore(store_);
  }

  ~QueryExecutorTester() = default;

  execution::Dag* InstallQuery() {
    std::string schemafile;
    std::string jsonfile;

    bool ok = flatbuffers::LoadFile(
      "../../fbs/query_plan.fbs", false, &schemafile);
    if (!ok) {
      LOG(ERROR) << "Load query_plan schema file failed.";
    }

    ok = flatbuffers::LoadFile(
      "../../conf/query_plan.template.json", false, &jsonfile);
    if (!ok) {
      LOG(ERROR) << "Load query_plan json file failed.";
    }

    flatbuffers::Parser parser;
    const char* include_paths[] = { "../../fbs/" };
    ok = parser.Parse(schemafile.c_str(), include_paths);
    if (!ok) {
      LOG(ERROR) << "Parse query_plan schema file failed.";
    }

    ok = parser.Parse(jsonfile.c_str());
    if (!ok) {
      LOG(ERROR) << "Parse query_plan json file failed.";
    }

    uint8_t* buf = parser.builder_.GetBufferPointer();
    auto query_plan_rep = GetQueryPlanRep(buf);
    auto* dag = new execution::Dag(query_plan_rep);
    return dag;
  }

  bool UninstallQuery(execution::Dag* dag) {
    delete dag;
    return true;
  }

  seastar::future<> RunQuery(execution::Dag* dag) {
    return seastar::do_with(execution::QueryExecutor(dag), [this] (auto& executor) {
      return executor.Execute(2, store_).then([] (QueryResponse&& res) {
        auto* results = res.GetRep()->results();
        for (auto* result : *results) {
          io::RecordBatchView view{result->value()};
          assert(view.Valid());
          LOG(INFO) << "Get query response: opid = " << result->opid()
                    << ", vid = " << result->vid()
                    << ", record num = " << view.RecordNum();
          EXPECT_EQ(view.RecordNum(), 2);
        }
      });
    });
  }

private:
  std::unique_ptr<storage::SampleStore> store_;
};

TEST(QueryExecutor, RunQuery) {
  InitGoogleLogging();

  auto* tester = new QueryExecutorTester();

  char prog_name[] = "query_executor_unittest";
  char docker_opt[] = "--thread-affinity=0";
  char cores[16];
  snprintf(cores, sizeof(cores), "-c%d", 2);

  int ac = 3;
  char* av[] = {prog_name, cores, docker_opt};

  hiactor::actor_app app;
  app.run(ac, av, [tester] () {
    execution::Dag* dag = tester->InstallQuery();
    return tester->RunQuery(dag).then([dag, tester] () {
      tester->UninstallQuery(dag);
      LOG(INFO) << "**** Test Query Executor Succeed!!! ****";
      hiactor::actor_engine().exit();
    });
  });
  delete tester;

  UninitGoogleLogging();
}
