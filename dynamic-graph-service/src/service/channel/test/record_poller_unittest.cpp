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

#include "cppkafka/utils/buffered_producer.h"
#include "seastar/core/loop.hh"

#include "service/actor_system.h"
#include "service/channel/record_poller.h"
#include "service/channel/sample_publisher.h"

using namespace dgs;
using namespace seastar;

static const int k_num_local_shards = 4;

class RecordPollerTester : public ::testing::Test {
public:
  RecordPollerTester() = default;
  ~RecordPollerTester() override = default;

protected:
  void SetUp() override {
    // Configure
    helper_.Initialize();

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
      "  producer-pool-size: 2\n";
    EXPECT_TRUE(Options::GetInstance().Load(options));
    EXPECT_TRUE(Schema::GetInstance().Init());

    // Init Kafaka producer
    KafkaProducerPool::GetInstance()->Init();
  }

  void TearDown() override {
    KafkaProducerPool::GetInstance()->Finalize();
    helper_.Finalize();
  }

  void PopulateFakeDataIntoKafka() {
    auto& opts = Options::GetInstance().GetRecordPollingOptions();
    cppkafka::BufferedProducer<std::string> producer(cppkafka::Configuration{
      {"metadata.broker.list", opts.FormatKafkaServers()},
      {"broker.address.family", "v4"}});
    cppkafka::MessageBuilder builder(opts.kafka_topic);
    for (int i = 0; i < k_num_local_shards; ++i) {
      auto record_batch = helper_.MakeRecordBatch(0, i, k_num_local_shards);
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


  InstallQueryRequest MakeInstallQueryRequest() {
    return helper_.MakeInstallQueryRequest();
  }

protected:
  TestHelper  helper_;
  ActorSystem actor_system_{WorkerType::Sampling, 0, 1, k_num_local_shards};
};


TEST_F(RecordPollerTester, RecordPollerFunctionality) {
  // populate fake data.
  PopulateFakeDataIntoKafka();
  auto* p_router = helper_.GetPartitionRouter().get();
  // sync meta and install query.
  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this, p_router] {
    // create actor reference to all shards.
    std::vector<SamplingActor_ref> refs;
    for (uint32_t i = 0; i < actor::GlobalShardCount(); ++i) {
      auto builder = hiactor::scope_builder(i);
      refs.push_back(MakeSamplingActorInstRef(builder));
    }

    auto req = MakeInstallQueryRequest();
    std::vector<PartitionId> pub_kafka_pids = {0, 1, 2, 3};
    auto payload = std::make_shared<SamplingInitPayload>(
        req.CloneBuffer(), helper_.GetSampleStore(), helper_.GetSampleBuilder(), helper_.GetSubsTable(),
        "hash", 4, "hash", 4, 4, p_router->GetRoutingInfo(), pub_kafka_pids);

    return seastar::do_with(std::move(refs), [this, payload] (std::vector<SamplingActor_ref>& refs) {
      return seastar::parallel_for_each(boost::irange(0u, actor::GlobalShardCount()),
          [&refs, payload] (uint32_t shard_id) {
        return refs[shard_id].ExecuteAdminOperation(
          AdminRequest(AdminOperation::INIT, payload)).discard_result();
      });
    });
  });

  fut.wait();

  RecordPollingManager manager;
  //FIXME(@LiSu) add configure for the mapping of kafka_partitions to poller
  std::vector<uint32_t> kafka_partitions = {0, 1, 2, 3};
  Partitioner partitioner = PartitionerFactory::Create("hash", 4);
  manager.Init(std::move(partitioner), p_router->GetRoutingInfo(), kafka_partitions, {});
  manager.Start();

  sleep(10);
  manager.Stop();
}
