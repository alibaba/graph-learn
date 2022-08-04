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

#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>

#include "common/log.h"
#include "core/io/record.h"
#include "core/io/record_builder.h"
#include "core/storage/sample_builder.h"

using namespace dgs;

void ParseRecord(io::RecordBuilder& record_builder, Schema& schema, std::string& s) {
  std::string delimiter = "\t";
  std::string piece;
  size_t last = 0;
  size_t next = 0;
  next = s.find(delimiter, last);
  piece = s.substr(last, next-last);
  last = next + 1;
  auto type = std::stoull(piece, nullptr, 10);

  AttributeType property_type = schema.GetAttrDefByName("timestamp").Type();
  AttributeValueType value_type = schema.GetAttrDefByName("timestamp").ValueType();
  // Parse vertex
  if (type == 0) {
    next = s.find(delimiter, last);
    piece = s.substr(last, next-last);
    last = next + 1;
    auto vid = std::stoll(piece, nullptr, 10);

    next = s.find(delimiter, last);
    piece = s.substr(last, next-last);
    last = next + 1;
    auto ts = std::stoul(piece, nullptr, 10);

    record_builder.Clear();
    auto attr = reinterpret_cast<int8_t*>(&ts);
    record_builder.AddAttribute(property_type, value_type, attr, sizeof(int32_t));
    record_builder.BuildAsVertexRecord(0, vid);
  }

  // Parse Edge
  if(type == 1) {
    next = s.find(delimiter, last);
    piece = s.substr(last, next-last);
    last = next + 1;
    auto src = std::stoll(piece, nullptr, 10);

    next = s.find(delimiter, last);
    piece = s.substr(last, next-last);
    last = next + 1;
    auto dst = std::stoll(piece, nullptr, 10);

    next = s.find(delimiter, last);
    piece = s.substr(last, next-last);
    last = next + 1;
    auto ts = std::stoul(piece, nullptr, 10);

    record_builder.Clear();
    auto attr = reinterpret_cast<int8_t*>(&ts);
    record_builder.AddAttribute(property_type, value_type, attr, sizeof(int32_t));
    record_builder.BuildAsEdgeRecord(1, 0, src, 0, dst);
  }
}

void PrepareRecordBatches(std::string file_path,
                          Schema& schema,
                          int batch_size,
                          std::vector<io::RecordBatch>& inputs) {
  io::RecordBuilder record_builder;
  io::RecordBatchBuilder batch_builder;
  std::ifstream infile(file_path);
  std::string s;
  int num = 0;
  while(std::getline(infile, s)) {
    ParseRecord(record_builder, schema, s);
    batch_builder.AddRecord(record_builder);
    if(batch_builder.RecordNum() == batch_size) {
      batch_builder.Finish();
      inputs.emplace_back(io::RecordBatch(act::BytesBuffer(
        reinterpret_cast<const char*>(batch_builder.BufPointer()),
        batch_builder.BufSize())));
      batch_builder.Clear();
    }
  }
  if(batch_builder.RecordNum() > 0) {
    batch_builder.Finish();
    inputs.emplace_back(io::RecordBatch(act::BytesBuffer(
      reinterpret_cast<const char*>(batch_builder.BufPointer()),
      batch_builder.BufSize())));
    batch_builder.Clear();
  }
  infile.close();
}

int main(int argc, char** argv) {
  std::string input_file_name = argv[1];
  int batch_size = std::atoi(argv[2]);
  int num_edge_sampler = std::atoi(argv[3]);

  std::string input_path = "../../../python/data/generated/" + input_file_name;

  auto& schema = Schema::GetInstance();
  bool ok = schema.Init("../../../conf/schema.template.json",
                        "../../../fbs/schema.fbs",
                        {"../../../fbs/"});

  std::cout<<"Finished loading schema"<<std::endl;
  std::vector<io::RecordBatch> batches;
  PrepareRecordBatches(input_path, schema, batch_size, batches);

  std::cout<<"Finished preparing record batches from input file"<<std::endl;

  std::vector<PartitionId> managed_pids;
  managed_pids.push_back(0);
  storage::SampleBuilder builder(managed_pids,
    PartitionerFactory::Create("hash", 1));

  ParamMap vertex_params = {
    {"vtype", 0}, {"versions", 10}, {"strategy", 0}
  };
  ParamMap edge_params = {
    {"etype", 1}, {"fanout", 10}, {"strategy", 0}
  };
  builder.AddVertexSamplerParams(vertex_params, 0);
  for(int oid = 0; oid < num_edge_sampler; oid++) {
    builder.AddEdgeSamplerParams(edge_params, oid);
  }

  auto start = std::chrono::steady_clock::now();
  size_t ret = 0;
  for (auto& batch : batches) {
    auto out = builder.Sample(batch);
    ret += out.size();
  }
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "Input file:" << input_file_name << "\n"
            << "Sample test finished, #samples = "<< ret << " in "
            << elapsed.count() << " seconds."<< std::endl;
}