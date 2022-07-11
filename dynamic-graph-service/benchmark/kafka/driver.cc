/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <iostream>
#include <numeric>
#include "driver.h"

namespace benchmark {
namespace kafka {

int64_t CurrentTimeInUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

Producer::Producer(int32_t partition_id) : partition_id_(partition_id) {
}

void Producer::Init() {
  cppkafka::Configuration conf{
    {"metadata.broker.list",  brokers},
    {"broker.address.family", "v4"},
    {"linger.ms", linger_ms},
    {"batch.num.messages", batch_msg_num}};
  conf.set_delivery_report_callback(
      [] (cppkafka::Producer& producer, const cppkafka::Message& msg) {
    if (msg.get_error()) {
      producer.produce(msg);
    }
  });
  producer_ = std::make_unique<cppkafka::Producer>(std::move(conf));
}

void Producer::Run() {
  thread_ = std::thread(&Producer::DoProduce, this);
}

void Producer::Join() {
  thread_.join();
}

void Producer::DoProduce() {
  Init();
  cppkafka::MessageBuilder builder(topic);
  builder.partition(partition_id_);
  char message[message_size];
  builder.payload({message, message_size});

  size_t partition_message_num = message_num / partition_num;
  start_ = CurrentTimeInUs();
  for (size_t i = 0; i < partition_message_num; i++) {
    while (true) {
      try {
        producer_->produce(builder);
        break;
      } catch (...) {
        Flush();
      }
    }
    producer_->poll(std::chrono::milliseconds(0));
  }
  Flush();
  end_ = CurrentTimeInUs();
}

void Producer::Flush() {
  while (true) {
    try {
      producer_->flush();
      break;
    } catch (...) {}
  }
}

Consumer::Consumer(int32_t partition_id) : partition_id_(partition_id) {
}

void Consumer::Init() {
  cppkafka::Configuration conf{
    {"metadata.broker.list", brokers},
    {"broker.address.family", "v4"},
    {"group.id", "kafka_benchmark_consumers"},
    {"enable.auto.commit", false}};
  consumer_ = std::make_unique<cppkafka::Consumer>(std::move(conf));
  auto offset = std::get<1>(consumer_->query_offsets(
      cppkafka::TopicPartition{topic, partition_id_}));
  if (workload_type == Consuming) {
    offset -= static_cast<int64_t>(message_num / partition_num);
  }
  std::cout << "consumer " << partition_id_
            << " will poll from offset " << offset << std::endl;
  consumer_->assign({cppkafka::TopicPartition{topic, partition_id_, offset}});
//  consumer_->set_timeout(std::chrono::milliseconds(0));
}

void Consumer::Run() {
  thread_ = std::thread(&Consumer::DoConsume, this);
}

void Consumer::Join() {
  thread_.join();
}

void Consumer::DoConsume() {
  size_t partition_message_num = message_num / partition_num;
  size_t polled_num = 0;
  start_ = CurrentTimeInUs();
  while (true) {
    if (polled_num >= partition_message_num) {
      break;
    }
    auto msg = consumer_->poll();
    if (msg && !msg.get_error()) {
      polled_num++;
    }
  }
  end_ = CurrentTimeInUs();
}

void Driver::RunWorkload(const bpo::variables_map& map) {
  SetOptions(map);
  std::cout << "-- workload_type: " << workload_type << std::endl;
  std::cout << "-- kafka brokers: " << brokers << std::endl;
  std::cout << "-- kafka topic: " << topic << std::endl;
  std::cout << "-- partition num: " << partition_num << std::endl;
  std::cout << "-- message size: " << message_size << std::endl;
  std::cout << "-- message num: " << message_num << std::endl;
  std::cout << "-- linger time in ms: " << linger_ms << std::endl;
  std::cout << "-- max batch message num: " << batch_msg_num << std::endl;
  std::vector<Producer> producers;
  producers.reserve(partition_num);
  if (workload_type == Producing || workload_type == Mixed) {
    for (int32_t i = 0; i < partition_num; i++) {
      producers.emplace_back(i);
    }
  }
  std::vector<Consumer> consumers;
  consumers.reserve(partition_num);
  if (workload_type == Consuming || workload_type == Mixed) {
    for (int32_t i = 0; i < partition_num; i++) {
      consumers.emplace_back(i);
    }
    for (auto& c : consumers) {
      c.Init();
    }
  }
  if (workload_type == Producing) {
    for (auto& p : producers) {
      p.Run();
    }
    for (auto& p : producers) {
      p.Join();
    }
  } else if (workload_type == Consuming) {
    for (auto& c : consumers) {
      c.Run();
    }
    for (auto& c : consumers) {
      c.Join();
    }
  } else if (workload_type == Mixed) {
    for (auto& p : producers) {
      p.Run();
    }
    for (auto& c : consumers) {
      c.Run();
    }
    for (auto& c : consumers) {
      c.Join();
    }
    for (auto& p : producers) {
      p.Join();
    }
  }

  auto size_in_mb = ((double)(message_num * message_size)) / (1024 * 1024);
  if (workload_type == Producing || workload_type == Mixed) {
    int64_t min_start = std::numeric_limits<int64_t>::max();
    int64_t max_end = std::numeric_limits<int64_t>::min();
    for (auto& p : producers) {
      min_start = std::min(min_start, p.Start());
      max_end = std::max(max_end, p.End());
    }
    auto slice_in_second = ((double)(max_end - min_start)) / (1000 * 1000);
    auto p_tp = size_in_mb / slice_in_second;
    std::cout << "Producing Throughput:" << p_tp << std::endl;
  }
  if (workload_type == Consuming || workload_type == Mixed) {
    int64_t min_start = std::numeric_limits<int64_t>::max();
    int64_t max_end = std::numeric_limits<int64_t>::min();
    for (auto& c : consumers) {
      min_start = std::min(min_start, c.Start());
      max_end = std::max(max_end, c.End());
    }
    auto slice_in_second = ((double)(max_end - min_start)) / (1000 * 1000);
    auto c_tp = size_in_mb / slice_in_second;
    std::cout << "Consuming Throughput:" << c_tp << std::endl;
  }
}

} // namespace kafka
} // namespace benchmark
