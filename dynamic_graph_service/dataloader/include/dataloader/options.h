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

#ifndef DATALOADER_OPTIONS_H_
#define DATALOADER_OPTIONS_H_

#include <cstdint>
#include <string>

namespace dgs {
namespace dataloader {

/// Options for data loader.
struct Options {
  static Options& Get() {
    static Options options;
    return options;
  }

  /// The brokers of output kafka queues.
  std::string output_kafka_brokers = "localhost:9092";

  /// The name of output kafka topic.
  std::string output_kafka_topic = "record-batches";

  /// The partition number of output kafka topic.
  uint32_t output_kafka_partitions = 1;

  /// The global data partitions for graph records.
  uint32_t data_partitions = 8;

private:
  Options() = default;
};

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_OPTIONS_H_
