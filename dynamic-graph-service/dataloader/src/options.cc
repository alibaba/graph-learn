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

#include "dataloader/options.h"
#include "dataloader/utils.h"

namespace dgs {
namespace dataloader {

bool Options::LoadFile(const std::string& option_file) {
  YAML::Node node;
  try {
    node = YAML::LoadFile(option_file);
  } catch (const YAML::ParserException& e) {
    LOG(ERROR) << "Loading malformed option file " << option_file << ": " << e.what();
    return false;
  } catch (const YAML::BadFile& e) {
    LOG(ERROR) << "Cannot load option file " << option_file << ": " << e.what();
    return false;
  }
  return SetOptions(node);
}

bool Options::SetOptions(const YAML::Node& map) {
  bool suc = true;
  suc &= SetScalarOption(fbs_file_dir, map, "option", "fbs-file-dir");
  suc &= SetScalarOption(schema_file, map, "option", "schema-file");
  suc &= SetScalarOption(coord_ipaddr, map, "option", "coord-ipaddr");
  suc &= SetScalarOption(coord_connect_timeout_sec, map, "option", "coord-connect-timeout-sec");
  suc &= SetScalarOption(coord_heartbeat_interval_sec, map, "option", "coord-heartbeat-interval-sec");
  suc &= SetScalarOption(loader_num, map, "option", "loader-num");
  suc &= SetScalarOption(loader_id, map, "option", "loader-id");
  suc &= SetScalarOption(data_partitions, map, "option", "data-partition-num");
  suc &= SetListOption(output_kafka_brokers, map, "option", "output-kafka-brokers");
  suc &= SetScalarOption(output_kafka_topic, map, "option", "output-kafka-topic");
  suc &= SetScalarOption(output_kafka_partitions, map, "option", "output-kafka-partitions");
  suc &= SetScalarOption(output_batch_size, map, "option", "output-batch-size");
  return suc;
}

std::string Options::FormatOutputKafkaBrokers() const {
  return StrJoin(output_kafka_brokers, ",");
}

}  // namespace dataloader
}  // namespace dgs
