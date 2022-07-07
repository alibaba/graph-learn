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

#ifndef DGS_SERVICE_TEST_TEST_HELPER_H_
#define DGS_SERVICE_TEST_TEST_HELPER_H_

#include <curl/curl.h>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "gtest/gtest.h"
#include "seastar/core/alien.hh"
#include "seastar/core/print.hh"
#include "seastar/core/sleep.hh"

#include "service/service.h"
#include "service/request/query_request.h"
#include "service/request/query_response.h"
#include "service/serving_group.actg.h"

namespace dgs {

using namespace seastar;
using namespace httpd;

class TestHelper {
public:
  TestHelper() {
    ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");
    ::system("rm -rf ./tmp_subs_table && mkdir -p ./tmp_subs_table");

    std::vector<PartitionId> pids{0, 1, 2, 3};
    store_ = std::make_unique<storage::SampleStore>(
        pids, PartitionerFactory::Create("hash", 4),
        "./tmp_store/", "./tmp_store/",
        storage::RdbEnv::Default());

    sample_builder_ = std::make_unique<storage::SampleBuilder>(
      pids, PartitionerFactory::Create("hash", 4));

    subs_table_ = std::make_unique<storage::SubscriptionTable>(
      pids, PartitionerFactory::Create("hash", 4),
        "./tmp_subs_table/", "./tmp_subs_table/",
        storage::RdbEnv::Default());

    subs_table_->SetDSWorkerPartitioner(4, "hash");

    partition_router_.reset(MakePartitionRouter());
  }

  virtual ~TestHelper() = default;

  storage::SampleStore* GetSampleStore() {
    return store_.get();
  }

  storage::SampleBuilder* GetSampleBuilder() {
    return sample_builder_.get();
  }

  storage::SubscriptionTable* GetSubsTable() {
    return subs_table_.get();
  }

  std::unique_ptr<PartitionRouter>& GetPartitionRouter() {
    return partition_router_;
  }

  void Initialize();
  void Finalize();

  // Make User Requests: InstallQuery, RunQuery
  InstallQueryRequest MakeInstallQueryRequest();
  RunQueryRequest MakeRunQueryRequest(VertexId vid);

  // Make RecordBatch contains one vertex and it's multiple neighbor edges.
  io::RecordBatch MakeRecordBatch(PartitionId pid, VertexId vid, size_t batch_size);

  // Make a fake samplestore.
  void MakeSampleStore();

  // Making partition routing info
  PartitionRouter* MakePartitionRouter();

  void PrintQueryResponse(const QueryResponse& res);

protected:
  VertexType vtype_ = 0;
  EdgeType   etype_ = 1;
  std::unique_ptr<PartitionRouter>            partition_router_;
  std::unique_ptr<storage::SampleStore>       store_;
  std::unique_ptr<storage::SampleBuilder>     sample_builder_;
  std::unique_ptr<storage::SubscriptionTable> subs_table_;
};

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

class ServiceTestHelper : public TestHelper {
public:
  ServiceTestHelper() = default;

  ~ServiceTestHelper() override = default;

  void InstallQuery(WorkerType worker_type);

  void SendRunQuery(VertexId vid);

public:
  std::vector<SamplingActor_ref>     sampling_refs_;
  std::vector<ServingActor_ref>      serving_refs_;
  std::vector<DataUpdateActor_ref>   data_update_refs_;
};

inline
void ServiceTestHelper::InstallQuery(WorkerType worker_type) {
  sampling_refs_.reserve(actor::LocalShardCount());
  serving_refs_.reserve(actor::LocalShardCount());
  data_update_refs_.reserve(actor::LocalShardCount());

  hiactor::scope_builder spl_builder(0);
  for (unsigned i = 0; i < actor::LocalShardCount(); i++) {
    auto g_sid = actor::GlobalShardIdAnchor() + i;
    spl_builder.set_shard(g_sid);
    sampling_refs_.emplace_back(MakeSamplingActorInstRef(spl_builder));
  }

  hiactor::scope_builder srv_builder(0, MakeServingGroupScope());
  for (unsigned i = 0; i < actor::LocalShardCount(); i++) {
    auto g_sid = actor::GlobalShardIdAnchor() + i;
    srv_builder.set_shard(g_sid);
    serving_refs_.emplace_back(MakeServingActorInstRef(srv_builder));
    data_update_refs_.emplace_back(MakeDataUpdateActorInstRef(srv_builder));
  }

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this, worker_type] {
    auto req = MakeInstallQueryRequest();
    if (worker_type == WorkerType::Sampling) {
      std::vector<PartitionId> pub_kafka_pids = {0, 1, 2, 3};
      auto payload = std::make_shared<SamplingInitPayload>(
          req.CloneBuffer(), store_.get(), sample_builder_.get(), subs_table_.get(),
          "hash", 4, "hash", 4, 4, partition_router_->GetRoutingInfo(), pub_kafka_pids);
      return seastar::parallel_for_each(boost::irange(0u, actor::LocalShardCount()), [this, payload] (uint32_t i) {
        return sampling_refs_[i].ExecuteAdminOperation(
          AdminRequest(AdminOperation::INIT, payload)).discard_result();
      });
    } else {
      auto payload = std::make_shared<ServingInitPayload>(
          req.CloneBuffer(), store_.get());
      return seastar::parallel_for_each(boost::irange(0u, actor::LocalShardCount()), [this, payload] (uint32_t i) {
        return serving_refs_[i].ExecuteAdminOperation(AdminRequest(AdminOperation::INIT, payload)).then([this, payload, i] (auto) {
          return data_update_refs_[i].ExecuteAdminOperation(AdminRequest(AdminOperation::INIT, payload)).discard_result();
        });
      });
    }
  });

  fut.wait();
}

inline
void TestHelper::Initialize() {
  // clear all existing storage first.
  ::system("rm -rf estore_* vstore_* subs_table* record_polling_offsets");
  InitGoogleLogging();
  FLAGS_alsologtostderr = true;

  Schema::GetInstance().Init();
}

inline
void TestHelper::Finalize() {
  UninitGoogleLogging();
}

inline
InstallQueryRequest TestHelper::MakeInstallQueryRequest() {
  std::string schemafile;
  std::string jsonfile;
  const char* default_schema = "../../fbs/install_query_req.fbs";
  const char* default_json = "../../conf/install_query_req_ut.json";
  bool ok;
  ok = flatbuffers::LoadFile(default_schema, false, &schemafile);
  if (!ok) { LOG(FATAL) << "Load install_query_request schema file failed.\n"; }
  ok = flatbuffers::LoadFile(default_json, false, &jsonfile);
  if (!ok) { LOG(FATAL) << "Load install_query_request json file failed.\n"; }

  flatbuffers::Parser parser;
  const char* include_paths[] = { "../../fbs/" };
  ok = parser.Parse(schemafile.c_str(), include_paths);
  if (!ok) { LOG(FATAL) << "Parse install_query_request schema file failed.\n"; }
  ok = parser.Parse(jsonfile.c_str());
  if (!ok) { LOG(FATAL) << "Parse install_query_request json file failed.\n"; }

  auto* ptr = reinterpret_cast<char*>(parser.builder_.GetBufferPointer());
  auto *rep = GetInstallQueryRequestRep(ptr);
  auto size = parser.builder_.GetSize();
  auto buf = actor::BytesBuffer(ptr, size);
  return InstallQueryRequest(std::move(buf), true);
}

inline
RunQueryRequest TestHelper::MakeRunQueryRequest(VertexId vid) {
  return {0, vid};
}

inline
io::RecordBatch TestHelper::MakeRecordBatch(PartitionId pid,
                                            VertexId vid,
                                            size_t batch_size) {
  AttributeType timestamp_type =
    Schema::GetInstance().GetAttrDefByName("timestamp").Type();

  io::RecordBatchBuilder batch_builder;
  io::RecordBuilder builder;
  int64_t timestamp = 1000;
  int32_t flipped = -10;

  batch_builder.SetStorePartitionId(pid);
  builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
      reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
  builder.BuildAsVertexRecord(vtype_, vid);
  batch_builder.AddRecord(builder);
  builder.Clear();

  for (VertexId dst_vid = vid + 1; dst_vid < batch_size; ++dst_vid) {
    timestamp += flipped * dst_vid;
    // add edge `vid -> dst_v`.
    builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
      reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
    builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid, dst_vid);
    batch_builder.AddRecord(builder);
    builder.Clear();
    flipped = -flipped;
  }

  batch_builder.Finish();

  auto *ptr = const_cast<char*>(reinterpret_cast<
      const char*>(batch_builder.BufPointer()));
  auto size = batch_builder.BufSize();
  auto buf = actor::BytesBuffer(ptr, size,
      seastar::make_object_deleter(std::move(batch_builder)));

  return io::RecordBatch(std::move(buf));
}

inline
void TestHelper::MakeSampleStore() {
  VertexId vid = 2;
  uint64_t fanout = 2;

  std::vector<OperatorId> vops = {2};
  std::vector<OperatorId> eops = {1, 3};
  io::RecordBuilder builder;

  AttributeType timestamp_type =
    Schema::GetInstance().GetAttrDefByName("timestamp").Type();
  int64_t timestamp = 1000;

  for (auto vop: vops) {
    for (uint64_t idx = 0; idx < fanout; ++idx) {
      storage::Key vkey{vtype_, vid, vop, idx};
      builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
        reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
      builder.BuildAsVertexRecord(vtype_, vid);
      store_->PutVertex(vkey, {builder.BufPointer(), builder.BufSize()});
      builder.Clear();
      timestamp += 1;
    }
  }

  for (auto eop: eops) {
    for (uint64_t idx = 0; idx < fanout; ++idx) {
      storage::Key ekey{vtype_, vid, eop, idx};
      builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
        reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
      builder.BuildAsEdgeRecord(etype_, vtype_, vtype_, vid, vid);
      store_->PutEdge(ekey, {builder.BufPointer(), builder.BufSize()});
      builder.Clear();
      timestamp += 1;
    }
  }
}

inline
PartitionRouter* TestHelper::MakePartitionRouter() {
  auto* router = new PartitionRouter();
  for (uint32_t i = 0; i < 4; i++) {
    auto update = RoutingUpdate(i, i);
    router->UpdatePartitionRoutingInfo(update);
  }
  return router;
}

void TestHelper::PrintQueryResponse(const QueryResponse& res) {
  auto* results = res.GetRep()->results();
  for (auto* result : *results) {
    AttributeType timestamp_type =
    Schema::GetInstance().GetAttrDefByName("timestamp").Type();

    io::RecordBatchView view{result->value()};
    assert(view.Valid());
    fmt::print("Get query response: opid={}, vid={}, #records={}\n",
               result->opid(), result->vid(), view.RecordNum());
    for (size_t idx = 0; idx < view.RecordNum(); ++idx) {
      auto rec_view = view.GetRecordByIdx(idx);
      if (rec_view.Type() == RecordType::VERTEX) {
        auto vertex = rec_view.AsVertexRecord();
        fmt::print("{}th record: vid={}, timestamp={}\n",
                   idx,
                   vertex.Id(),
                   vertex.LookUpAttrByType(timestamp_type).AsInt64());
      } else {
        auto edge = rec_view.AsEdgeRecord();
        fmt::print("{}th record: src_vid={}, dst_vid={}, timestamp={}\n",
                   idx,
                   edge.SrcId(), edge.DstId(),
                   edge.LookUpAttrByType(timestamp_type).AsInt64());
      }
    }
  }
}

inline
void ServiceTestHelper::SendRunQuery(VertexId vid) {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;
  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "localhost:10000/serving/w0?qid=0&vid=2");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    EXPECT_TRUE(res == 0);
    actor::BytesBuffer ret(readBuffer.size());
    auto data = ret.get_write();
    std::memcpy(data, readBuffer.data(), readBuffer.size());
    curl_easy_cleanup(curl);
    PrintQueryResponse(QueryResponse(std::move(ret)));
  }
}

} // namespace dgs

#endif // DGS_SERVICE_TEST_TEST_HELPER_H_