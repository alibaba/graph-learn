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

#include "boost/program_options.hpp"
#include "cppkafka/consumer.h"

#include "dataloader/schema.h"
#include "dataloader/fbs/record_generated.h"

namespace bpo = boost::program_options;

std::string schema_json_file = "../../../conf/schema.e2e.json";

std::string GetVertexTypeName(dgs::dataloader::VertexType vtype) {
  auto& schema = dgs::dataloader::Schema::Get();
  return schema.GetVertexDefByType(vtype).Name();
}

std::string GetEdgeTypeName(dgs::dataloader::EdgeType etype) {
  auto& schema = dgs::dataloader::Schema::Get();
  return schema.GetEdgeDefByType(etype).Name();
}

std::string GetAttrTypeName(dgs::dataloader::AttributeType type) {
  auto& schema = dgs::dataloader::Schema::Get();
  return schema.GetAttrDefByType(type).Name();
}

std::string GetProperty(const dgs::AttributeRecordRep* rep) {
  auto value_type = rep->value_type();
  if (value_type == dgs::AttributeValueTypeRep_INT32) {
    auto value = *reinterpret_cast<const int32_t*>(rep->value_bytes()->data());
    return std::to_string(value);
  } else if (value_type == dgs::AttributeValueTypeRep_INT64) {
    auto value = *reinterpret_cast<const int64_t*>(rep->value_bytes()->data());
    return std::to_string(value);
  } else if (value_type == dgs::AttributeValueTypeRep_FLOAT32) {
    auto value = *reinterpret_cast<const float*>(rep->value_bytes()->data());
    return std::to_string(value);
  } else if (value_type == dgs::AttributeValueTypeRep_FLOAT64) {
    auto value = *reinterpret_cast<const double*>(rep->value_bytes()->data());
    return std::to_string(value);
  } else if (value_type == dgs::AttributeValueTypeRep_STRING) {
    return {reinterpret_cast<const char*>(rep->value_bytes()->data()), rep->value_bytes()->size()};
  }
  return "###";
}

void PrintAttrsInfo(const flatbuffers::Vector<flatbuffers::Offset<dgs::AttributeRecordRep>>* attrs) {
  for (auto a : *attrs) {
    std::cout << "(" << GetAttrTypeName(a->attr_type()) << ": " << GetProperty(a) << ")";
  }
}

void PrintVertexInfo(const dgs::VertexRecordRep* rep) {
  std::cout << "[Vertex: " << GetVertexTypeName(rep->vtype()) << "(" << rep->vid() << ")] ";
  PrintAttrsInfo(rep->attributes());
  std::cout << std::endl;
}

void PrintEdgeInfo(const dgs::EdgeRecordRep* rep) {
  std::cout << "[Edge: "
            << GetVertexTypeName(rep->src_vtype()) << "(" << rep->src_id() << ") -> "
            << GetEdgeTypeName(rep->etype()) << " -> "
            << GetVertexTypeName(rep->dst_vtype()) << "(" << rep->dst_id() << ")] ";
  PrintAttrsInfo(rep->attributes());
  std::cout << std::endl;
}

void PrintRecordBatchInfo(const cppkafka::Message& msg) {
  std::cout << "-- Record Batch At Offset " << msg.get_offset() << std::endl;
  auto* rep = flatbuffers::GetRoot<dgs::RecordBatchRep>(msg.get_payload().get_data());
  std::cout << "[Data Partition: " << rep->partition() << "]" << std::endl;
  for (auto* record: *rep->records()) {
    auto r_type = record->record_type();
    if (r_type == dgs::RecordUnionRep_VertexRecordRep) {
      PrintVertexInfo(record->record_as_VertexRecordRep());
    } else if (r_type == dgs::RecordUnionRep_EdgeRecordRep) {
      PrintEdgeInfo(record->record_as_EdgeRecordRep());
    }
  }
}

void PollWith(cppkafka::Consumer* consumer) {
  std::chrono::milliseconds timeout(100);
  uint32_t failed_times = 0;
  while (true) {
    if (failed_times >= 3) {
      break;
    }
    auto msg = consumer->poll(timeout);
    if (!msg || msg.get_error()) {
      failed_times++;
    } else {
      PrintRecordBatchInfo(msg);
      failed_times = 0;
    }
  }
}

int main(int argc, char** argv) {
  bpo::options_description options("Output Kafka Record Viewer Options");
  options.add_options()
    ("output-kafka-brokers,b", bpo::value<std::string>(), "the output kafka brokers")
    ("output-kafka-topic,t", bpo::value<std::string>(), "the output kafka topic")
    ("output-kafka-partition,p", bpo::value<int32_t>(), "the output kafka topic partition id")
    ("start-offset,o", bpo::value<int64_t>()->default_value(0), "start offset of topic record viewer")
    ("schema-json-file,s", bpo::value<std::string>(), "graph schema json file");
  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  } catch (...) {
    std::cerr << "Undefined options in command line." << std::endl;
    return -1;
  }
  bpo::notify(vm);

  if (!vm.count("output-kafka-brokers")) {
    std::cerr << "The kafka brokers must be specified." << std::endl;
    return -1;
  }
  std::string brokers = vm["output-kafka-brokers"].as<std::string>();

  if (!vm.count("output-kafka-topic")) {
    std::cerr << "The kafka topic must be specified." << std::endl;
  }
  std::string topic = vm["output-kafka-topic"].as<std::string>();

  if (vm.count("output-kafka-partition")) {
    std::cerr << "The kafka partition id must be specified." << std::endl;
  }
  int32_t pid = vm["output-kafka-partition"].as<int32_t>();

  int64_t start_offset = 0;
  if (vm.count("start-offset")) {
    start_offset = vm["start-offset"].as<int64_t>();
  }

  if (vm.count("schema-json-file")) {
    schema_json_file = vm["schema-json-file"].as<std::string>();
  }

  std::ifstream in_file(schema_json_file);
  std::string schema_json((std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());

  dgs::dataloader::Schema::Get().Init(schema_json);

  cppkafka::Consumer consumer(cppkafka::Configuration{
    {"metadata.broker.list", brokers},
    {"broker.address.family", "v4"},
    {"group.id", "output-viewer"},
    {"enable.auto.commit", false}});
  consumer.assign({cppkafka::TopicPartition{topic, pid, start_offset}});
  PollWith(&consumer);

  return 0;
}
