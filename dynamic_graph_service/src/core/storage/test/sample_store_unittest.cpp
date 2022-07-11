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

#include "gtest/gtest.h"

#include "common/log.h"
#include "core/io/record_builder.h"
#include "core/storage/sample_store.h"

using namespace dgs;

class SampleStoreTester {
public:
  SampleStoreTester() = default;
  ~SampleStoreTester() = default;

  void Run() {

    ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store && mkdir -p ./tmp_store/restore");

    auto partitioner = PartitionerFactory::Create("hash", 4);
    std::vector<PartitionId> pids{0, 1, 2, 3};
    auto store = std::make_unique<storage::SampleStore>(
        pids, std::move(partitioner), "./tmp_store/",
        "./tmp_store/", storage::RdbEnv::Default());

    storage::Key vkey1{1, 2, 2, 0};
    storage::Key vkey2{1, 2, 2, 1};

    io::RecordBuilder builder;
    builder.BuildAsVertexRecord(1, 11);
    EXPECT_TRUE(store->PutVertex(vkey1, {builder.BufPointer(), builder.BufSize()}));

    builder.Clear();
    builder.BuildAsVertexRecord(2, 22);
    EXPECT_TRUE(store->PutVertex(vkey2, {builder.BufPointer(), builder.BufSize()}));

    io::Record record;
    EXPECT_TRUE(store->GetVertex(vkey1, &record));

    storage::Key::Prefix pkey{1, 2, 2};
    std::vector<storage::KVPair> records;
    EXPECT_TRUE(store->GetVerticesByPrefix(pkey, &records));

    EXPECT_TRUE(store->DeleteVertex(vkey1));
    EXPECT_FALSE(store->GetVertex(vkey1, &record));

    auto bk_infos = store->Backup();
    EXPECT_EQ(bk_infos.size(), 4);

    auto restore_store = std::make_unique<storage::SampleStore>(
      bk_infos, PartitionerFactory::Create("hash", 4), "./tmp_store/restore/",
      "./tmp_store/", storage::RdbEnv::Default());
    
    io::Record restore_record;
    EXPECT_FALSE(restore_store->GetVertex(vkey1, &restore_record));
    EXPECT_TRUE(restore_store->GetVertex(vkey2, &restore_record));
  }
};

TEST(SampleStore, StorageFunctionality) {
  InitGoogleLogging();

  SampleStoreTester tester;
  tester.Run();

  UninitGoogleLogging();
}
