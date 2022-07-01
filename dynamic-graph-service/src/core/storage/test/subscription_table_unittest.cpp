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
#include "core/storage/subscription_table.h"

using namespace dgs;

TEST(SubscriptionTable, Functionality) {
  // clear all existing storage first.
  ::system("rm -rf estore_* vstore_* subs_table* record_polling_offsets");
  ::system("rm -rf ./tmp_subs_table && mkdir -p ./tmp_subs_table && mkdir -p ./tmp_subs_table/restore");

  InitGoogleLogging();

  auto tbl_partitioner = PartitionerFactory::Create("hash", 4);
  std::vector<PartitionId> pids{0, 1, 2, 3};
  auto table = std::make_unique<storage::SubscriptionTable>(
      pids, std::move(tbl_partitioner), "./tmp_subs_table/",
      "./tmp_subs_table/", storage::RdbEnv::Default());

  table->SetDSWorkerPartitioner(4, "hash");

  std::vector<io::SubsRule> subs_rules;
  io::SubsRule rule{0, 1, 10, 0};
  for (int i = 0; i < 4; ++i) {
    rule.worker_id = i;
    subs_rules.push_back(rule);
  }

  std::vector<uint32_t> output1;
  table->UpdateRules(subs_rules, &output1);

  std::vector<storage::SubsInfo> output2;
  table->GetSubscribedWorkers(0, rule.pkey, &output2);

  EXPECT_EQ(output1.size(), 4);
  EXPECT_EQ(output2.size(), 4);

  auto bk_infos = table->Backup();
  EXPECT_EQ(bk_infos.size(), 4);

  auto restore_table = std::make_unique<storage::SubscriptionTable>(
      bk_infos, PartitionerFactory::Create("hash", 4), "./tmp_subs_table/restore/",
      "./tmp_subs_table/", storage::RdbEnv::Default());
  restore_table->SetDSWorkerPartitioner(4, "hash");

  std::vector<storage::SubsInfo> output3;
  restore_table->GetSubscribedWorkers(0, rule.pkey, &output3);
  EXPECT_EQ(output3.size(), 4);

  UninitGoogleLogging();
}
