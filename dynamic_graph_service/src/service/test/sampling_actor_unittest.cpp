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

#include "cppkafka/consumer.h"

#include "core/execution/query_executor.h"
#include "core/io/sample_update_batch.h"

using namespace dgs;

class SamplingActorModuleTest : public ::testing::Test {
public:
  SamplingActorModuleTest() : helper_(4, 2, 4, 4) {}
  ~SamplingActorModuleTest() override = default;

protected:
  void SetUp() override {
    InitGoogleLogging();
    FLAGS_alsologtostderr = true;

    // Configure for SamplePublisher.
    std::string options =
        "sample-publishing:\n"
        "  output-kafka-servers:\n"
        "    - localhost:9092\n"
        "  kafka-topic: sampling-actor-ut\n"
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

  static void ConsumeSamplingOutputAndVerifyCorrectness(
      int expected_msg_num,
      int32_t expected_update_num,
      int32_t kafka_partition) {
    auto& opts = Options::GetInstance().GetSamplePublishingOptions();
    auto kafka_servers = opts.FormatKafkaServers();
    auto sampling_output_topic = opts.kafka_topic;

    cppkafka::Consumer consumer(cppkafka::Configuration{
            {"metadata.broker.list", kafka_servers},
            {"group.id", "graph_update_record_pollers"},
            {"enable.auto.commit", false}});
    consumer.assign({cppkafka::TopicPartition{
      sampling_output_topic, kafka_partition, 0}});
    consumer.set_timeout(std::chrono::milliseconds(2000));

    int32_t num_incoming_updates = 0;
    while (--expected_msg_num >= 0) {
      auto msg = consumer.poll();
      EXPECT_TRUE(!msg == false);
      EXPECT_TRUE(!msg.get_error());

      auto data  = msg.get_payload().get_data();
      auto data_size = msg.get_payload().get_size();
      auto buf = actor::BytesBuffer(
        const_cast<char*>(reinterpret_cast<const char*>(data)),
        data_size, seastar::make_object_deleter(std::move(msg)));

      auto updates = io::SampleUpdateBatch::Deserialize(std::move(buf));
      num_incoming_updates += static_cast<int32_t>(updates.size());
    }
    EXPECT_TRUE(num_incoming_updates == expected_update_num);
  }

protected:
  SamplingTestHelper helper_;
};

TEST_F(SamplingActorModuleTest, RunAll) {
  // sync metadata with global coordinator.
  // serving worker num: 4, partitioning stratety: hash
  // serving storage partition num: 4，strategy: hash
  //
  // install query.
  // query plan is as follows:
  // all vertex types(vtype = 0) are same, so we ignore vtype below.
  // nodes: Source: 0, ESampler: 1, VSampler: 2, ESampler: 3
  // edges: 0 -> 1; 0 -> 2; 1 -> 2; 1 -> 3;
  // ESampler 1 type: TOPK_BY_TIMESTAMP, K = 2;
  // ESampler 2 type: TOPK_BY_TIMESTAMP, K = 2;
  helper_.InstallQuery();

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this] {
    // apply graph updates.
    // number of vertices: 4
    // record batch 0: (vertex: 0, edges: 0 -> 1; 0 -> 2; 0 -> 3)
    // record batch 1: (vertex: 1, edges: 1 -> 2; 1 -> 3)
    // record batch 2: (vertex: 2, edges: 2 -> 3)
    // record batch 3: (vertex: 3, edges:)
    uint32_t num_v = 4;
    uint32_t num_p = 4;
    return seastar::parallel_for_each(
        boost::irange(0u, num_v), [num_v, num_p, this] (uint32_t i) {
      VertexId vid = i;
      auto pid = i % num_p;
      auto shard_id = pid % actor::GlobalShardCount();
      return helper_.GetSamplingActorRef(shard_id).ApplyGraphUpdates(
          SamplingTestHelper::MakeRecordBatch(pid, vid, num_v)).discard_result();
    }).then([] {
      // sample updates for vertex 0:
      // - batch 1 (size = 3): (VSampler2: vertex 0) (ESampler1: 0 -> 1) (ESampler1: 0 -> 2)
      // - batch 2 (size = 3): (VSampler2: vertex 1) (ESampler3: 1 -> 2) (ESampler1: 1 -> 3)
      // - batch 2 (size = 2): (VSampler2: vertex 2) (ESampler3: 2 -> 3)
      // sample updates for vertex 1:
      // - batch 1 (size = 3): (VSampler2: vertex 1) (ESampler1: 1 -> 2) (ESampler1: 1 -> 3)
      // - batch 2 (size = 2): (VSampler2: vertex 2) (ESampler3: 2 -> 3)
      // - batch 2 (size = 1): (VSampler2: vertex 3)
      // sample updates for vertex 2:
      // - batch 1 (size = 2): (VSampler2: vertex 2) (ESampler1: 2 -> 3)
      // - batch 2 (size = 1): (VSampler2: vertex 3)
      // sample updates for vertex 3:
      // - batch 1 (size = 1): (VSampler2: vertex 3)

      // kafka p0: 1 message, size = [3, 3, 2]
      ConsumeSamplingOutputAndVerifyCorrectness(3, 8, 0);
      // kafka p1: 2 messages, sizes = [3, 2, 1]
      ConsumeSamplingOutputAndVerifyCorrectness(3, 6, 1);
      // kafka p2: 3 messages, sizes = [2, 1]
      ConsumeSamplingOutputAndVerifyCorrectness(2, 3, 2);
      // kafka p3: 3 messages, sizes = [1]
      ConsumeSamplingOutputAndVerifyCorrectness(1, 1, 3);
    });
  });
  fut.wait();
}
