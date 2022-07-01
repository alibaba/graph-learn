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

#include "yaml-cpp/yaml.h"

#include "dataloader/logging.h"

namespace dgs {
namespace dataloader {

template <typename T>
inline
bool SetScalarOption(T& target, const YAML::Node& map, const std::string& group, const std::string& option) {
  auto node = map[option];
  if (node) {
    if (!node.IsScalar()) {
      LOG(ERROR) << "Wrong type with input " << group << ":" << option
      << ", scalar type required.";
      return false;
    }
    try {
      target = node.as<T>();
    } catch (const YAML::TypedBadConversion<T>& e) {
      LOG(ERROR) << "Type conversion error with input "
      << group << ":" << option << ", " << e.what();
      return false;
    }
  }
  return true;
}

template <typename T>
inline
bool SetListOption(std::vector<T>& target, const YAML::Node& map, const std::string& group, const std::string& option) {
  auto node = map[option];
  if (node) {
    if (!node.IsSequence()) {
      LOG(ERROR) << "Wrong type with input " << group << ":" << option
      << ", sequence type required.";
      return false;
    }
    std::vector<T> list(node.size());
    for (size_t i = 0; i < node.size(); i++) {
      try {
        list[i] = node[i].as<T>();
      } catch (const YAML::TypedBadConversion<T>& e) {
        LOG(ERROR) << "Type conversion error with input "
        << group << ":" << option << ", " << e.what();
        return false;
      }
    }
    target = std::move(list);
  }
  return true;
}

/// Options for data loader.
struct Options {
  static Options& GetInstance();

  /// Load options from yaml file.
  /// \return @True if successful.
  bool LoadFile(const std::string& option_file);

  /// Specify the directory which contains all .fbs files
  std::string fbs_file_dir = "../../../fbs";

  /// Specify the json file of graph schema
  std::string schema_file = "../../../conf/schema.template.json";

  /// The ip address and port of global coordinator.
  std::string coord_ipaddr = "0.0.0.0:50051";

  /// The timeout in seconds when connect to coordinator.
  uint32_t coord_connect_timeout_sec = 60;

  /// The heartbeat interval in seconds for reporting stats to coordinator.
  uint32_t coord_heartbeat_interval_sec = 60;

  /// The total data loader number and current data loader id.
  /// These two options will be set from coordinator when service is inited.
  uint32_t loader_num = 1;
  uint32_t loader_id = 0;

  /// The global data partitions for graph records.
  uint32_t data_partitions = 8;

  /// The brokers of output kafka queues.
  std::vector<std::string> output_kafka_brokers = { "localhost:9092" };
  std::string FormatOutputKafkaBrokers() const;

  /// The name of output kafka topic.
  std::string output_kafka_topic = "record-batches";

  /// The partition number of output kafka topic.
  uint32_t output_kafka_partitions = 2;

  /// The max record number in an output record batch.
  uint32_t output_batch_size = 16;

private:
  Options() = default;
  bool SetOptions(const YAML::Node& map);
};

inline
Options& Options::GetInstance() {
  static Options options;
  return options;
}

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_OPTIONS_H_
