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

#include "service/test/test_helper.h"
#include "service/channel/record_poller.h"
#include "cppkafka/utils/buffered_producer.h"

using namespace dgs;

static const int k_num_local_shards = 2;

class RecordPollerTester : public ::testing::Test {
public:
  RecordPollerTester() : helper_(4, k_num_local_shards, 4, 4) {}
  ~RecordPollerTester() override = default;

protected:
  void SetUp() override {
    InitGoogleLogging();
    FLAGS_alsologtostderr = true;

    std::string options =
      "worker-type: Sampling\n"
      "record-polling:\n"
      "  source-kafka-servers:\n"
      "    - localhost:9092\n"
      "  kafka-topic: record-poller-ut-1\n"
      "  kafka-partition-num: 4\n"
      "sample-publishing:\n"
      "  output-kafka-servers:\n"
      "    - localhost:9092\n"
      "  kafka-topic: record-poller-ut-2\n"
      "  kafka-partition-num: 4\n"
      "  producer-pool-size: 1\n";
    EXPECT_TRUE(Options::GetInstance().Load(options));
    EXPECT_TRUE(Schema::GetInstance().Init());

    helper_.Initialize();

    // Init Kafka producer
    KafkaProducerPool::GetInstance()->Init();
  }

  void TearDown() override {
    KafkaProducerPool::GetInstance()->Finalize();
    UninitGoogleLogging();
  }

  static void PopulateFakeDataIntoKafka() {
    auto& opts = Options::GetInstance().GetRecordPollingOptions();
    cppkafka::BufferedProducer<std::string> producer(cppkafka::Configuration{
      {"metadata.broker.list", opts.FormatKafkaServers()},
      {"broker.address.family", "v4"}});
    cppkafka::MessageBuilder builder(opts.kafka_topic);
    for (int i = 0; i < k_num_local_shards; ++i) {
      auto record_batch = SamplingTestHelper::MakeRecordBatch(0, i, k_num_local_shards);
      try {
        builder.partition(i).payload(
          {record_batch.Data(), record_batch.Size()});
        producer.sync_produce(builder);
      } catch (const cppkafka::HandleException& e) {
        LOG(ERROR) << "Populate fake data into kafka error "
          << e.get_error().to_string();
      }
    }
  }

protected:
  SamplingTestHelper helper_;
};


TEST_F(RecordPollerTester, RecordPollerFunctionality) {
  // install query
  helper_.InstallQuery();

  // populate fake data.
  PopulateFakeDataIntoKafka();

  RecordPollingManager manager;
  std::vector<uint32_t> kafka_partitions = {0, 1, 2, 3};
  Partitioner partitioner = PartitionerFactory::Create("hash", 4);
  auto* p_router = helper_.GetPartitionRouter();
  manager.Init(std::move(partitioner), p_router->GetRoutingInfo(), kafka_partitions, {});
  manager.Start();

  sleep(10);
  manager.Stop();
}
