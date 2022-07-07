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

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "gtest/gtest.h"
#include "hiactor/core/actor-app.hh"

#include "common/log.h"
#include "core/io/record_builder.h"
#include "core/io/sample_update_batch.h"
#include "service/actor_ref_builder.h"
#include "service/request/query_request.h"
#include "service/serving_group.actg.h"

using namespace dgs;
using namespace seastar;

InstallQueryRequest MakeInstallQueryRequest() {
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

class DataUpdateActorTester {
public:
  DataUpdateActorTester() = default;
  ~DataUpdateActorTester() = default;

  void Run() {
    ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");
    auto partitioner = PartitionerFactory::Create("hash", 1);
    std::vector<PartitionId> pids{0};
    store_ = std::make_unique<storage::SampleStore>(
        pids, std::move(partitioner),
        "./tmp_store/", "./tmp_store/",
        storage::RdbEnv::Default());

    char  arg0[] = "DataUpdateActorTester";
    char  arg1[] = "-c2";
    char* argv[] = {&arg0[0], &arg1[0]};
    hiactor::actor_app sys;
    sys.run(2, argv, [this] {
      hiactor::scope_builder builder(0, MakeServingGroupScope());
      auto actor_ref = MakeDataUpdateActorInstRef(builder);

      io::RecordBuilder record_builder;
      storage::Key key(0, 0, 0, 0);
      int64_t timestamp = 1000;
      auto attr = reinterpret_cast<int8_t*>(&timestamp);
      record_builder.AddAttribute(0, AttributeValueType::INT64,
        attr, sizeof(int64_t));
      record_builder.BuildAsVertexRecord(0, 0);
      const uint8_t* buf = record_builder.BufPointer();
      auto size = record_builder.BufSize();
      actor::BytesBuffer tp(reinterpret_cast<const char*>(buf), size);
      io::Record record(std::move(tp));
      std::vector<const storage::KVPair*> pairs;
      storage::KVPair pair(key, std::move(record));
      pairs.emplace_back(&pair);
      io::SampleUpdateBatch batch(0, pairs);
      EXPECT_TRUE(batch.GetUpdatesNum() == 1);

      auto req = MakeInstallQueryRequest();
      auto payload = std::make_shared<ServingInitPayload>(
          req.CloneBuffer(), store_.get());

      return actor_ref.ExecuteAdminOperation(AdminRequest(AdminOperation::INIT, payload)).then(
        [actor_ref, batch=std::move(batch), this] (auto) mutable {
          return actor_ref.Update(std::move(batch)).then([this] (actor::Void ret) {
            storage::Key key(0, 0, 0, 0);
            io::Record record;
            EXPECT_TRUE(store_->GetVertex(key, &record));
            EXPECT_TRUE(record.GetView().AsVertexRecord()
              .GetAttrByIdx(0).AsInt64() == 1000);
            hiactor::actor_engine().exit();
          });
      });
    });
  }

private:
  std::unique_ptr<storage::SampleStore> store_;
};

TEST(DataUpdateActor, DataUpdateFunctionality) {
  InitGoogleLogging();
  FLAGS_alsologtostderr = true;

  DataUpdateActorTester tester;
  tester.Run();

  UninitGoogleLogging();
}