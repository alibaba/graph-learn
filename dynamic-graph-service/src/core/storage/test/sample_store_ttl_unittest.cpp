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

class SpecialTimeEnv : public rocksdb::EnvWrapper {
public:
  explicit SpecialTimeEnv(Env* base) : EnvWrapper(base) {
    base->GetCurrentTime(&current_time_);
  }

  void Sleep(int64_t sleep_time) { 
    current_time_ += sleep_time;
  }

  rocksdb::Status GetCurrentTime(int64_t* current_time) override {
    *current_time = current_time_;
    return rocksdb::Status::OK();
  }

private:
  int64_t current_time_ = 0;
};

class SampleStoreTtlTester {
public:
  SampleStoreTtlTester() {
      env_ = new SpecialTimeEnv(storage::RdbEnv::Default());
  };
  ~SampleStoreTtlTester() = default;

  void Run() {

    ::system("rm -rf ./tmp_store && mkdir -p ./tmp_store");

    auto partitioner = PartitionerFactory::Create("hash", 4);
    std::vector<PartitionId> pids{0, 1, 2, 3};
    auto store = std::make_unique<storage::SampleStore>(
        pids, std::move(partitioner), "./tmp_store",
        "./tmp_store", env_);

    storage::Key ekey1{2, 2, 2, 0};
    storage::Key ekey2{2, 2, 2, 1};

    io::RecordBuilder builder;
    builder.BuildAsEdgeRecord(4, 2, 2, 1, 2);
    EXPECT_TRUE(store->PutEdge(ekey1, {builder.BufPointer(), builder.BufSize()}));

    builder.Clear();
    builder.BuildAsEdgeRecord(4, 2, 2, 2, 3);
    EXPECT_TRUE(store->PutEdge(ekey2, {builder.BufPointer(), builder.BufSize()}));

    dgs::io::Record record;
    EXPECT_TRUE(store->GetEdge(ekey2, &record));

    storage::Key::Prefix pkey{2, 2, 2};
    std::vector<storage::KVPair> ret;
    EXPECT_TRUE(store->GetEdgesByPrefix(pkey, &ret));

    env_->Sleep(3601);

    io::Record record_expired;
    EXPECT_TRUE(store->GetEdge(ekey2, &record_expired));

    // Trigger compactions
    for (uint64_t i = 3; i < 2000000; ++i) {
      storage::Key ekey3{2, 2, 2, i};
      builder.Clear();
      builder.BuildAsEdgeRecord(4, 2, 2, i, i + 1);
      store->PutEdge(ekey3, {builder.BufPointer(), builder.BufSize()});
    }

    io::Record record_compacted;
    EXPECT_FALSE(store->GetEdge(ekey2, &record_compacted));
  }

private:
  SpecialTimeEnv* env_;
};

TEST(SampleStoreTtl, StorageTtlFunctionality) {
  // clear all existing storage first.
  ::system("rm -rf estore_* vstore_* subs_table* record_polling_offsets");

  InitGoogleLogging();

  SampleStoreTtlTester tester;
  tester.Run();

  UninitGoogleLogging();
}
