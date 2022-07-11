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
  SamplingActorModuleTest() : helper_(4, 4, 2, 4, 4) {}
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

    // Init Kafaka producer
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

      io::SampleUpdateBatch su_batch{std::move(buf)};

      auto updates = su_batch.GetSampleUpdates();
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
  // nodes: Source: 0, Sampler: 1, Traverser: 2, Sampler: 3
  // edges: 0 -> 1; 0 -> 2; 1 -> 3;
  // Sampler 1 type: TOPK_BY_TIMESTAMP, K = 2;
  // Sampler 2 type: TOPK_BY_TIMESTAMP, K = 2;
  helper_.InstallQuery();

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this] {
    uint32_t num_v = 4;
    uint32_t num_p = 4;
    // apply graph updates.
    // number of vertices: 4
    // record batch on shard 0: vertex: 0, edges: 0 -> 1, 0 -> 2; 0 -> 3;
    // record batch on shard 1: vertex: 1, edges: 1 -> 2; 1 -> 3;
    // record batch on shard 2: vertex: 2, edges: 2 -> 3;
    // record batch on shard 3: vertex: 3, edges: ;
    //
    // stage 1: sampled batch size for each shard -> Stored in SampleStore.
    // sampled batch on shard 0: 5, 1 for vertex, 4 for edges(2 for each sampler);
    // sampled batch on shard 1: 5, 1 for vertex, 4 for edges(2 for each sampler);
    // sampled batch on shard 2: 3, 1 for vertex, 2 for edges(1 for each sampler);
    // sampled batch on shard 3: 1, 1 for vertex;
    //
    // stage 2: current subscription information for each shard -> Send to Kafka by SamplePublisher.
    // subs_info size on shard 0: 3, indices in sampled batch = [0, 1, 3], dst_worker_id = 0;
    // subs_info size on shard 1: 3, indices in sampled batch = [0, 1, 3], dst_worker_id = 1;
    // subs_info size on shard 2: 2, indices in sampled batch = [0, 1], dst_worker_id = 2;
    // subs_info size on shard 3: 1, indices in sampled batch = [0], dst_worker_id = 3;
    //
    // stage 3: downstream subscription rules using query dependency info
    // rule buffer on shard 0: #rules=3, {vid = [1, 2], op_id=(1<<32)+2&op_id=3}, subscribed worker id: 0;
    // rule buffer on shard 1: #rules=2, {vid = [2, 3], op_id=(1<<32)+2&&op_id=3}, subscribed worker id: 1;
    // rule buffer on shard 2: #rules=1, {vid = [3], op_id=(1<<32)+2&&op_id=3}, subscribed worker id: 2;
    // rule buffer on shard 3: #rules=0;
    // in detail,
    // --- the rule buffer on shard 0 is seperated into 4 piece:
    // piece 1: rule = {vid = 1, op_id = (1<<32)+2, subs_worker_id = 0; }, dst global shard id: 1
    // piece 2: rule = {vid = 1, op_id = 3, subs_worker_id = 0; }, dst global shard id: 1
    // piece 3: rule = {vid = 2, op_id = (1<<32)+2, subs_worker_id = 0; }, dst global shard id: 2
    // piece 4: rule = {vid = 2, op_id = 3, subs_worker_id = 0; }, dst global shard id: 2
    // --- the rule buffer on shard 1 is seperated into 4 piece:
    // piece 1: rule = {vid = 2, op_id = (1<<32)+2, subs_worker_id = 1; }, dst global shard id: 2
    // piece 2: rule = {vid = 2, op_id = 3, subs_worker_id = 1; }, dst global shard id: 2
    // piece 3: rule = {vid = 3, op_id = (1<<32)+2, subs_worker_id = 1; }, dst global shard id: 3
    // piece 4: rule = {vid = 3, op_id = 3, subs_worker_id = 1; }, dst global shard id: 3
    // --- the rule buffer on shard 2 is seperated into 2 piece:
    // piece 3: rule = {vid = 3, op_id = (1<<32)+2, subs_worker_id = 2; }, dst global shard id: 3
    // piece 4: rule = {vid = 3, op_id = 3, subs_worker_id = 2; }, dst global shard id: 3
    // - the rule buffer on shard 3 has no rule.

    // for the remaining stages, we will discuss rule behaviors triggered by rule buffer on shard 0 only.
    //
    // stage 4: update subscription rules and publish new subscribed records.
    // rule received(piece 1, op_id = (1<<32)+2(VSampler)) on shard id 1: new collected sample batch size: 1;
    // rule received(piece 2, op_id = 3(ESampler)) on shard id 1: new collected sample batch size: 2(ref to stage 1);
    // rule received(piece 3, op_id = (1<<32)+2(VSampler)) on shard id 2: new collected sample batch size: 1;
    // rule received(piece 4, op_id = 3(ESampler)) on shard id 2: new collected sample batch size: 1(ref to stage 1);
    //
    // end of all stages.

    // all the non-empty collected new sample batch will be send to Kafka by SamplePublisher.
    return seastar::parallel_for_each(
        boost::irange(0u, num_v), [num_v, num_p, this] (uint32_t i) {
      VertexId vid = i;
      auto pid = vid % num_p;
      auto shard_id = vid % actor::GlobalShardId();
      return helper_.GetSamplingActorRef(shard_id).ApplyGraphUpdates(
          SamplingTestHelper::MakeRecordBatch(pid, vid, num_v)).discard_result();
    }).then([] {
      // kafka p0: 1 message, size = [3]
      ConsumeSamplingOutputAndVerifyCorrectness(1, 3, 0);
      // kafka p1: 2 messages, sizes = [3, 3]
      ConsumeSamplingOutputAndVerifyCorrectness(2, 6, 1);
      // kafka p2: 3 messages, sizes = [2, 2, 2]
      ConsumeSamplingOutputAndVerifyCorrectness(3, 6, 2);
      // kafka p3: 3 messages, sizes = [1, 1, 1]
      ConsumeSamplingOutputAndVerifyCorrectness(3, 3, 3);
    });
  });
  fut.wait();
}
