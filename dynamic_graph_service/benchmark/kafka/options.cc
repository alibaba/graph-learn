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

#include "options.h"

namespace benchmark {
namespace kafka {

WorkloadType workload_type = Producing;

std::string brokers = "localhost:9092";
std::string topic = "kafka_benchmark";
int32_t partition_num = 1;

size_t message_size = 128;
size_t message_num = 1000;

size_t linger_ms = 10;
size_t batch_msg_num = 10000;

void SetOptions(const bpo::variables_map& map) {
  if (map.count("workload")) {
    auto token = map["workload"].as<std::string>();
    if (token == "producing" || token == "p") {
      workload_type = Producing;
    } else if (token == "consuming" || token == "c") {
      workload_type = Consuming;
    } else if (token == "mixed" || token == "m") {
      workload_type = Mixed;
    } else {
      throw std::runtime_error("Invalid workload type");
    }
  }
  if (map.count("brokers")) {
    brokers = map["brokers"].as<std::string>();
  }
  if (map.count("topic")) {
    topic = map["topic"].as<std::string>();
  }
  if (map.count("partition-num")) {
    partition_num = map["partition-num"].as<int32_t>();
  }
  if (map.count("message-size")) {
    message_size = map["message-size"].as<size_t>();
  }
  if (map.count("message-num")) {
    message_num = map["message-num"].as<size_t>();
  }
  if (map.count("linger-ms")) {
    linger_ms = map["linger-ms"].as<size_t>();
  }
  if (map.count("batch-num-messages")) {
    batch_msg_num = map["batch-num-messages"].as<size_t>();
  }
}

} // namespace kafka
} // namespace benchmark
