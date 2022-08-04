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

static const VertexType test_vtype_ = 0;
static const EdgeType   test_etype_ = 4;

class BasicTestHelper {
public:
  BasicTestHelper() = default;
  virtual ~BasicTestHelper() = default;

  static InstallQueryRequest MakeInstallQueryRequest();
  static void PrintQueryResponse(const QueryResponse& res);

protected:
  static std::vector<PartitionId> MakePidVector(uint32_t num_partitions);
};

inline
InstallQueryRequest BasicTestHelper::MakeInstallQueryRequest() {
  std::string schemafile;
  std::string jsonfile;
  const char* default_schema = "../../fbs/install_query_req.fbs";
  const char* default_json = "../../conf/ut/install_query.ut.json";
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
//  auto *rep = GetInstallQueryRequestRep(ptr);
  auto size = parser.builder_.GetSize();
  auto buf = act::BytesBuffer(ptr, size);
  return InstallQueryRequest(std::move(buf), true);
}

inline
void BasicTestHelper::PrintQueryResponse(const QueryResponse& res) {
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
                   vertex.LookUpAttrByType(timestamp_type).AsValue<int64_t>());
      } else {
        auto edge = rec_view.AsEdgeRecord();
        fmt::print("{}th record: src_vid={}, dst_vid={}, timestamp={}\n",
                   idx,
                   edge.SrcId(), edge.DstId(),
                   edge.LookUpAttrByType(timestamp_type).AsValue<int64_t>());
      }
    }
  }
}

inline
std::vector<PartitionId>
BasicTestHelper::MakePidVector(uint32_t num_partitions) {
  std::vector<PartitionId> pids;
  pids.resize(num_partitions);
  for (uint32_t i = 0; i < num_partitions; i++) {
    pids[i] = i;
  }
  return pids;
}

class SamplingTestHelper : public BasicTestHelper {
public:
  SamplingTestHelper(uint32_t num_store_partitions,
                     uint32_t num_local_shards,
                     uint32_t num_pub_kafka_partitions,
                     uint32_t num_ds_worker)
    : num_store_partitions_(num_store_partitions),
      num_local_shards_(num_local_shards),
      num_pub_kafka_partitions_(num_pub_kafka_partitions),
      num_ds_worker_(num_ds_worker) {}
  ~SamplingTestHelper() override = default;

  void Initialize();

  void InstallQuery();

  // make test record batch
  static io::RecordBatch MakeRecordBatch(PartitionId pid,
                                         VertexId vid,
                                         size_t batch_size);

  SamplingActor_ref& GetSamplingActorRef(size_t idx) {
    return sampling_act_refs_.at(idx);
  }

  PartitionRouter* GetPartitionRouter() {
    return partition_router_.get();
  }

private:
  const uint32_t num_store_partitions_;
  const uint32_t num_local_shards_;
  const uint32_t num_pub_kafka_partitions_;
  const uint32_t num_ds_worker_;
  std::unique_ptr<PartitionRouter>            partition_router_;
  std::unique_ptr<storage::SampleStore>       store_;
  std::unique_ptr<storage::SampleBuilder>     sample_builder_;
  std::unique_ptr<storage::SubscriptionTable> subs_table_;
  std::unique_ptr<ActorSystem>       actor_sys_;
  std::vector<SamplingActor_ref>     sampling_act_refs_;
};

inline
void SamplingTestHelper::Initialize() {
  ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");
  ::system("rm -rf ./tmp_subs_table && mkdir -p ./tmp_subs_table");

  partition_router_ = std::make_unique<PartitionRouter>();
  for (uint32_t i = 0; i < num_store_partitions_; i++) {
    auto update = RoutingUpdate(i, i % num_local_shards_);
    partition_router_->UpdatePartitionRoutingInfo(update);
  }

  auto pids = MakePidVector(num_store_partitions_);
  sample_builder_ = std::make_unique<storage::SampleBuilder>(
      pids, PartitionerFactory::Create("hash", num_store_partitions_));
  store_ = std::make_unique<storage::SampleStore>(
      pids,
      PartitionerFactory::Create("hash", num_store_partitions_),
      "./tmp_store/",
      "./tmp_store/",
      storage::RdbEnv::Default());
  subs_table_ = std::make_unique<storage::SubscriptionTable>(
      pids,
      PartitionerFactory::Create("hash", num_store_partitions_),
      "./tmp_subs_table/",
      "./tmp_subs_table/",
      storage::RdbEnv::Default());
  subs_table_->SetDSWorkerPartitioner(num_ds_worker_, "hash");

  actor_sys_ = std::make_unique<ActorSystem>(
      WorkerType::Sampling, 0, 1, num_local_shards_);
  sampling_act_refs_.reserve(num_local_shards_);
  hiactor::scope_builder spl_builder;
  for (unsigned i = 0; i < num_local_shards_; i++) {
    auto g_sid = act::GlobalShardIdAnchor() + i;
    spl_builder.set_shard(g_sid);
    sampling_act_refs_.emplace_back(MakeSamplingActorInstRef(spl_builder));
  }
}

inline
void SamplingTestHelper::InstallQuery() {
  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [this] {
        auto req = MakeInstallQueryRequest();
        std::vector<uint32_t> kafka_to_wid;
        kafka_to_wid.resize(num_pub_kafka_partitions_);
        for (uint32_t i = 0; i < num_pub_kafka_partitions_; i++) {
          kafka_to_wid[i] = i % num_ds_worker_;
        }
        auto payload = std::make_shared<SamplingInitPayload>(
            req.CloneBuffer(),
            store_.get(),
            sample_builder_.get(),
            subs_table_.get(),
            "hash",
            num_store_partitions_,
            partition_router_->GetRoutingInfo(),
            num_ds_worker_,
            kafka_to_wid);
        return seastar::parallel_for_each(boost::irange(0u, num_local_shards_),
            [this, payload] (uint32_t i) {
          return sampling_act_refs_[i].ExecuteAdminOperation(
              AdminRequest(AdminOperation::INIT, payload)).discard_result();
        });
      });
  fut.wait();
}

inline
io::RecordBatch SamplingTestHelper::MakeRecordBatch(PartitionId pid,
                                                    VertexId vid,
                                                    size_t batch_size) {
  AttributeType timestamp_type =
      Schema::GetInstance().GetAttrDefByName("timestamp").Type();

  io::RecordBatchBuilder batch_builder;
  io::RecordBuilder builder;
  int64_t timestamp = 1000;
  int32_t flipped = -10;

  batch_builder.SetStorePartitionId(pid);
  builder.AddAttribute(
      timestamp_type, AttributeValueType::INT64,
      reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
  builder.BuildAsVertexRecord(test_vtype_, vid);
  batch_builder.AddRecord(builder);
  builder.Clear();

  for (VertexId dst_vid = vid + 1; dst_vid < batch_size; ++dst_vid) {
    timestamp += flipped * dst_vid;
    // add edge `vid -> dst_v`.
    builder.AddAttribute(
        timestamp_type, AttributeValueType::INT64,
        reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
    builder.BuildAsEdgeRecord(
        test_etype_, test_vtype_, test_vtype_, vid, dst_vid);
    batch_builder.AddRecord(builder);
    builder.Clear();
    flipped = -flipped;
  }

  batch_builder.Finish();

  auto *ptr = const_cast<char*>(reinterpret_cast<
      const char*>(batch_builder.BufPointer()));
  auto size = batch_builder.BufSize();
  auto buf = act::BytesBuffer(
      ptr, size, seastar::make_object_deleter(std::move(batch_builder)));

  return io::RecordBatch(std::move(buf));
}

class ServingTestHelper : public BasicTestHelper {
public:
  ServingTestHelper(uint32_t num_store_partitions,
                    uint32_t num_local_shards)
  : num_store_partitions_(num_store_partitions),
    num_local_shards_(num_local_shards) {}
  ~ServingTestHelper() override = default;

  void Initialize();

  void InstallQuery();

  // Make a fake sample store.
  void MakeSampleStore();

  static void SendRunQuery(VertexId vid);

  ServingActor_ref& GetServingActorRef(size_t idx) {
    return serving_act_refs_.at(idx);
  }

  DataUpdateActor_ref& GetDataUpdateActorRef(size_t idx) {
    return data_update_act_refs_.at(idx);
  }

  static RunQueryRequest MakeRunQueryRequest(VertexId vid) {
    return {0, vid};
  }

private:
  const uint32_t num_store_partitions_;
  const uint32_t num_local_shards_;
  std::unique_ptr<PartitionRouter>       partition_router_;
  std::unique_ptr<storage::SampleStore>  store_;
  std::unique_ptr<ActorSystem>      actor_sys_;
  std::vector<ServingActor_ref>     serving_act_refs_;
  std::vector<DataUpdateActor_ref>  data_update_act_refs_;
};

inline
void ServingTestHelper::Initialize() {
  ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");

  partition_router_ = std::make_unique<PartitionRouter>();
  for (uint32_t i = 0; i < num_store_partitions_; i++) {
    auto update = RoutingUpdate(i, i % num_local_shards_);
    partition_router_->UpdatePartitionRoutingInfo(update);
  }

  auto pids = MakePidVector(num_store_partitions_);
  store_ = std::make_unique<storage::SampleStore>(
      pids,
      PartitionerFactory::Create("hash", num_store_partitions_),
      "./tmp_store/",
      "./tmp_store/",
      storage::RdbEnv::Default());

  actor_sys_ = std::make_unique<ActorSystem>(
      WorkerType::Serving, 0, 1, num_local_shards_);
  serving_act_refs_.reserve(num_local_shards_);
  data_update_act_refs_.reserve(num_local_shards_);
  hiactor::scope_builder srv_builder(0, MakeServingGroupScope());
  for (unsigned i = 0; i < num_local_shards_; i++) {
    auto g_sid = act::GlobalShardIdAnchor() + i;
    srv_builder.set_shard(g_sid);
    serving_act_refs_.emplace_back(MakeServingActorInstRef(srv_builder));
    data_update_act_refs_.emplace_back(MakeDataUpdateActorInstRef(srv_builder));
  }
}

inline
void ServingTestHelper::InstallQuery() {
  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [this] {
        auto req = MakeInstallQueryRequest();
        auto payload = std::make_shared<ServingInitPayload>(
            req.CloneBuffer(), store_.get());
        return seastar::parallel_for_each(boost::irange(0u, num_local_shards_),
            [this, payload] (uint32_t i) {
          return serving_act_refs_[i].ExecuteAdminOperation(
            AdminRequest(AdminOperation::INIT, payload)
          ).then([this, payload, i] (auto) {
            return data_update_act_refs_[i].ExecuteAdminOperation(
                AdminRequest(AdminOperation::INIT, payload)).discard_result();
          });
        });
      });
  fut.wait();
}

inline
void ServingTestHelper::MakeSampleStore() {
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
      storage::Key vkey{test_vtype_, vid, vop, idx};
      builder.AddAttribute(
          timestamp_type, AttributeValueType::INT64,
          reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
      builder.BuildAsVertexRecord(test_vtype_, vid);
      store_->PutVertex(vkey, {builder.BufPointer(), builder.BufSize()});
      builder.Clear();
      timestamp += 1;
    }
  }

  for (auto eop: eops) {
    for (uint64_t idx = 0; idx < fanout; ++idx) {
      storage::Key ekey{test_vtype_, vid, eop, idx};
      builder.AddAttribute(
          timestamp_type, AttributeValueType::INT64,
          reinterpret_cast<int8_t*>(&timestamp), sizeof(int64_t));
      builder.BuildAsEdgeRecord(
          test_etype_, test_vtype_, test_vtype_, vid, vid);
      store_->PutEdge(ekey, {builder.BufPointer(), builder.BufSize()});
      builder.Clear();
      timestamp += 1;
    }
  }
}

inline
size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

inline
void ServingTestHelper::SendRunQuery(VertexId vid) {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;
  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "localhost:10000/serving/w0?qid=0&vid=2");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    EXPECT_TRUE(res == 0);
    act::BytesBuffer ret(readBuffer.size());
    auto data = ret.get_write();
    std::memcpy(data, readBuffer.data(), readBuffer.size());
    curl_easy_cleanup(curl);
    PrintQueryResponse(QueryResponse(std::move(ret)));
  }
}

} // namespace dgs

#endif // DGS_SERVICE_TEST_TEST_HELPER_H_