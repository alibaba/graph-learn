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
#include "common/options.h"

using namespace dgs;

void TestOptionsLoad() {
  std::string yaml_str =
    "worker-type: Sampling\n"
    "fbs-file-dir: path-a\n"
    "schema-file: path-b\n"
    "sample-store:\n"
    "  in-memory-mode: True\n"
    "  db-path: ./\n"
    "  backup-path: ./\n"
    "  ttl-hours: 4\n"
    "subscription-table:\n"
    "  ttl-hours: 2\n"
    "record-polling:\n"
    "  source-kafka-servers:\n"
    "    - server-1\n"
    "    - server-2\n"
    "sample-publishing:\n"
    "  producer-pool-size: 4\n"
    "logging:\n"
    "  rule-log-period: 10\n"
    "event-handler:\n"
    "  http-port: 1234\n";
  auto& opts = Options::GetInstance();
  EXPECT_TRUE(opts.Load(yaml_str));
  EXPECT_EQ(opts.GetWorkerType(), WorkerType::Sampling);
  EXPECT_EQ(opts.GetFbsFileDir(), "path-a");
  EXPECT_EQ(opts.GetSchemaFile(), "path-b");

  auto& sample_store_opts = opts.GetSampleStoreOptions();
  EXPECT_TRUE(sample_store_opts.in_memory_mode);
  EXPECT_EQ(sample_store_opts.db_path, "./");
  EXPECT_EQ(sample_store_opts.backup_path, "./");
  EXPECT_EQ(sample_store_opts.ttl_in_hours, 4);

  auto& subs_table_opts = opts.GetSubscriptionTableOptions();
  EXPECT_EQ(subs_table_opts.ttl_in_hours, 2);

  auto& poll_opts = opts.GetRecordPollingOptions();
  EXPECT_EQ(poll_opts.source_kafka_servers.size(), 2);
  EXPECT_EQ(poll_opts.source_kafka_servers[0], "server-1");
  EXPECT_EQ(poll_opts.source_kafka_servers[1], "server-2");
  EXPECT_EQ(poll_opts.FormatKafkaServers(), "server-1,server-2");

  auto& sample_pub_opts = opts.GetSamplePublishingOptions();
  EXPECT_EQ(sample_pub_opts.producer_pool_size, 4);

  auto& sampling_opts = opts.GetLoggingOptions();
  EXPECT_EQ(sampling_opts.rule_log_period, 10);

  auto& event_hdl_opts = opts.GetEventHandlerOptions();
  EXPECT_EQ(event_hdl_opts.http_port, 1234);
}

void TestOptionsLoadFile() {
  std::string options_file = "../../conf/serving_service_ut_options.yml";
  auto& opts = Options::GetInstance();
  EXPECT_TRUE(opts.LoadFile(options_file));
  EXPECT_EQ(opts.GetWorkerType(), WorkerType::Serving);
  EXPECT_EQ(opts.GetSampleStoreOptions().memtable_rep, "hashskiplist");
  EXPECT_TRUE(opts.GetSampleStoreOptions().in_memory_mode);
  EXPECT_EQ(opts.GetSampleStoreOptions().block_cache_capacity, 67108864);
  EXPECT_EQ(opts.GetRecordPollingOptions().thread_num, 1);
}

TEST(Options, Functionality) {
  InitGoogleLogging();
  TestOptionsLoad();
  TestOptionsLoadFile();
  UninitGoogleLogging();
}
