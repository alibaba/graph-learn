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

#include "cppkafka/consumer.h"
#include "gtest/gtest.h"
#include "hiactor/core/actor-app.hh"

#include "common/log.h"
#include "core/io/record_builder.h"
#include "service/channel/sample_publisher.h"

using namespace dgs;
using namespace seastar;

class SamplePublisherTester {
public:
  SamplePublisherTester() = default;
  ~SamplePublisherTester() = default;

  void Run() {
    KafkaProducerPool::GetInstance()->Init();

    char  arg0[] = "SamplePublisherTester";
    char  arg1[] = "--open-thread-resource-pool=true";
    char  arg2[] = "-c2";
    char* argv[] = {&arg0[0], &arg1[0], &arg2[0]};
    std::string topic = "sample-publisher-ut";
    hiactor::actor_app sys;
    sys.run(3, argv, [&topic] {
      auto* publisher = new SamplePublisher(topic, 4);
      io::RecordBuilder record_builder;
      int64_t timestamp = 1000;
      auto attr = reinterpret_cast<int8_t*>(&timestamp);
      record_builder.AddAttribute(0, AttributeValueType::INT64,
        attr, sizeof(int64_t));
      record_builder.BuildAsVertexRecord(0, 0);
      const uint8_t* buf = record_builder.BufPointer();
      auto size = record_builder.BufSize();
      actor::BytesBuffer tp(reinterpret_cast<const char*>(buf), size);
      io::Record record(std::move(tp));

      storage::Key key(0, 0, 0, 0);
      storage::KVPair pair(key, std::move(record));
      storage::SubsInfo info(0, 0);

      std::vector<storage::KVPair> batch;
      std::vector<storage::SubsInfo> infos;

      batch.emplace_back(std::move(pair));
      infos.emplace_back(std::move(info));

      std::vector<uint32_t> sink_kafka_partitions = {0, 1, 2, 3};
      publisher->UpdateSinkKafkaPartitions(sink_kafka_partitions, "hash", 4, 1);

      return publisher->Publish(batch, infos).then_wrapped([&topic] (auto&& f) {
        EXPECT_TRUE(!f.failed());
        cppkafka::Consumer consumer(cppkafka::Configuration{
          {"metadata.broker.list", "localhost:9092"},
          {"group.id", "graph_update_record_pollers"},
          {"enable.auto.commit", false}});
        uint32_t partition = 0;
        consumer.assign({cppkafka::TopicPartition{
          topic, static_cast<int32_t>(partition), 0}});
        consumer.set_timeout(std::chrono::milliseconds(1000));

        auto msg = consumer.poll();
        EXPECT_TRUE(msg);
        EXPECT_TRUE(!msg.get_error());

        auto data  = msg.get_payload().get_data();
        auto data_size = msg.get_payload().get_size();
        auto buf = actor::BytesBuffer(
          const_cast<char*>(reinterpret_cast<const char*>(data)),
          data_size, seastar::make_object_deleter(std::move(msg)));

        uint32_t expected_record_num = 1;
        uint32_t record_num;
        auto offset = buf.get() + sizeof(PartitionId);
        std::memcpy(&record_num, offset, sizeof(uint32_t));
        EXPECT_EQ(record_num, expected_record_num);
        offset += sizeof(uint32_t);

        storage::Key key(1, 1, 1, 1);
        storage::Key key_expected(0, 0, 0, 0);
        std::memcpy(&key, offset, sizeof(storage::Key));
        EXPECT_TRUE(std::memcmp(&key, &key_expected, sizeof(storage::Key)) == 0);
        offset += sizeof(storage::Key);
      }).then([publisher] {
        hiactor::actor_engine().exit();
        delete publisher;
      });
    });
    KafkaProducerPool::GetInstance()->Finalize();
  }
};

TEST(SamplePublisher, SamplePublishFunctionality) {
  InitGoogleLogging();

  SamplePublisherTester tester;
  tester.Run();

  UninitGoogleLogging();
}
