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
#include "driver.h"

using namespace benchmark::kafka;

int main(int argc, char** argv) {
  bpo::options_description options("Kafka Benchmark Options");
  options.add_options()
    ("workload,w", bpo::value<std::string>()->default_value("serial"), "workload type of benchmarking")
    ("brokers", bpo::value<std::string>()->default_value("localhost:9092"), "kafka brokers")
    ("topic", bpo::value<std::string>()->default_value("kafka_benchmark"), "kafka topic")
    ("partition-num,p", bpo::value<int32_t>()->default_value(1), "kafka partition number")
    ("message-size,s", bpo::value<size_t>()->default_value(128), "byte size of kafka message")
    ("message-num,n", bpo::value<size_t>()->default_value(1000), "message number of benchmarking")
    ("linger-ms,l", bpo::value<size_t>()->default_value(10), "linger timer in ms of producing")
    ("batch-num-messages,b", bpo::value<size_t>()->default_value(10000), "max message number in a kafka request");
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  } catch (...) {
    std::cout << "undefined options in command line." << std::endl;
    return 0;
  };
  bpo::notify(vm);

  Driver::RunWorkload(vm);
}
