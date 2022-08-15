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

#include "gtest/gtest.h"

#include "common/log.h"
#include "core/storage/sample_builder.h"
#include "core/io/record.h"
#include "core/io/record_builder.h"

using namespace dgs;

namespace dgs {

class SampleBuilderTester {
public:
  SampleBuilderTester() = default;
  ~SampleBuilderTester() = default;

  void Run() {
    auto& schema = Schema::GetInstance();
    bool ok = schema.Init("../../conf/ut/schema.ut.json",
                          "../../fbs/schema.fbs",
                          {"../../fbs/"});
    EXPECT_TRUE(ok);

    std::vector<PartitionId> managed_pids;
    managed_pids.push_back(0);

    storage::SampleBuilder builder(managed_pids,
      PartitionerFactory::Create("hash", 1));

    EdgeType etype = 3;
    ParamMap params = {
      {"etype", etype}, {"fanout", 2}, {"strategy", 0}
    };
    OperatorId oid = 0;

    builder.AddEdgeSamplerParams(params, oid);

    // Sample vertices
    VertexType vtype = 1;
    VertexId vid = 11;
    uint32_t index = 1000000;
    bool accepted;

    AttributeType timestamp_type = schema.GetAttrDefByName("timestamp").Type();

    int64_t timestamps[] = {1000, 900, 1100, 800};
    bool expected_accept[] = {true, true, true, false};
    uint32_t expected_index[] = {0, 1, 1, 1};
    io::RecordBuilder record_builder;
    for (auto timestamp : timestamps) {
      record_builder.Clear();
      auto attr = reinterpret_cast<int8_t*>(&timestamp);
      record_builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
        attr, sizeof(int64_t));
      record_builder.BuildAsVertexRecord(vtype, vid);
      const uint8_t* buf = record_builder.BufPointer();
      auto size = record_builder.BufSize();
      act::BytesBuffer tp(reinterpret_cast<const char*>(buf), size);
      io::Record record(std::move(tp));
      // accepted = builder.SampleVertex(record.GetView().AsVertexRecord(), index);
      // EXPECT_TRUE(accepted == expected_accept[i]);
      // if (accepted) {
      //   EXPECT_TRUE(index == expected_index[i]);
      // }
    }

    // Sample edges
    VertexType src_vtype = 1;
    VertexType dst_vtype = 1;
    VertexId src_id = 11;
    VertexId dst_id = 12;

    for (auto timestamp : timestamps) {
      record_builder.Clear();
      auto attr = reinterpret_cast<int8_t*>(&timestamp);
      record_builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
        attr, sizeof(int64_t));
      record_builder.BuildAsEdgeRecord(etype, src_vtype, dst_vtype, src_id, dst_id);
      const uint8_t* buf = record_builder.BufPointer();
      auto size = record_builder.BufSize();
      act::BytesBuffer tp(reinterpret_cast<const char*>(buf), size);
      io::Record record(std::move(tp));
      std::unordered_map<OperatorId, uint32_t> output;
      // accepted = builder.SampleEdge(record.GetView().AsEdgeRecord(), output);
      // EXPECT_TRUE(accepted == expected_accept[i]);
      // if (accepted) {
      //   EXPECT_TRUE(output.at(oid) == expected_index[i]);
      // }
    }

    // Dump the sampling state in sample builder as a checkpoint file
    const char* ckp_file = "./sample_builer_checkpoint.tmp";
    std::ofstream outfile(ckp_file);
    builder.Dump(outfile);

    // Load the sampling state in sample builder back from a checkpoint file
    storage::SampleBuilder builder_copy(managed_pids,
      PartitionerFactory::Create("hash", 1));

    builder_copy.AddEdgeSamplerParams(params, oid);
    std::ifstream infile(ckp_file);
    builder_copy.Load(infile);

    // Sample vertex using builder_copy
    record_builder.Clear();
    int64_t timestamp = 1200;
    auto attr = reinterpret_cast<int8_t*>(&timestamp);
    record_builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
      attr, sizeof(int64_t));
    record_builder.BuildAsVertexRecord(vtype, vid);
    const uint8_t* buf = record_builder.BufPointer();
    auto size = record_builder.BufSize();
    act::BytesBuffer tp(reinterpret_cast<const char*>(buf), size);
    io::Record record(std::move(tp));
    // accepted = builder_copy.SampleVertex(record.GetView().AsVertexRecord(), index);
    // EXPECT_TRUE(accepted);
    // EXPECT_TRUE(index == 0);

    // Sample edge using builder_copy
    record_builder.Clear();
    timestamp = 1200;
    attr = reinterpret_cast<int8_t*>(&timestamp);
    record_builder.AddAttribute(timestamp_type, AttributeValueType::INT64,
      attr, sizeof(int64_t));
    record_builder.BuildAsEdgeRecord(etype, src_vtype, dst_vtype, src_id, dst_id);
    const uint8_t* buf_copy = record_builder.BufPointer();
    auto size_copy = record_builder.BufSize();
    act::BytesBuffer tp_copy(reinterpret_cast<const char*>(buf_copy), size_copy);
    io::Record record_copy(std::move(tp_copy));
    std::unordered_map<OperatorId, uint32_t> output_copy;
    // accepted = builder_copy.SampleEdge(record_copy.GetView().AsEdgeRecord(), output_copy);
    // EXPECT_TRUE(accepted);
    // EXPECT_TRUE(index == 0);

    //Delete the tmp ccheckpoint file
    std::remove(ckp_file);
  }
};

}  // namespace dgs

TEST(SampleBuilder, SampleFunctionality) {
  InitGoogleLogging();
  SampleBuilderTester tester;
  tester.Run();
  UninitGoogleLogging();
}
