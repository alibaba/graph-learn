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

#include "core/io/record_builder.h"
#include "core/io/record.h"
#include "service/request/query_response.h"

using namespace dgs;
using namespace seastar;

void PrintQueryResponse(const QueryResponse& res) {
  auto* results = res.GetRep()->results();
  for (auto* result : *results) {
    io::RecordBatchView view{result->value()};
    assert(view.Valid());
    std::cout << "Get query response: "
              << "opid: " << result->opid()
              << ", vid: " << result->vid()
              << ", record num: " << view.RecordNum() << std::endl;
    for (size_t idx = 0; idx < view.RecordNum(); ++idx) {
      auto rec_view = view.GetRecordByIdx(idx);
      if (rec_view.Type() == RecordType::VERTEX) {
        auto vertex = rec_view.AsVertexRecord();
        std::cout << idx << "th record has vid: " << vertex.Id()
                  << ", attr 0: " << vertex.GetAttrByIdx(0).AsString() << std::endl;
      } else {
        auto edge = rec_view.AsEdgeRecord();
        std::cout << idx << "th record has srcid: " << edge.SrcId()
                  << ", dstid: " << edge.DstId()
                  << ", attr 0: " << edge.GetAttrByIdx(0).AsString() << std::endl;
      }
    }
  }
}

TEST(QueryReponse, PutGet) {
  QueryResponseBuilder res_builder;
  std::vector<io::Record> records;

  io::RecordBuilder builder1;
  std::string attr1 = "attr1";
  builder1.AddAttribute(0, AttributeValueType::STRING,
                       reinterpret_cast<const int8_t*>(attr1.data()), attr1.size());
  builder1.BuildAsVertexRecord(1, 11);

  auto *data1 = const_cast<char*>(reinterpret_cast<const char*>(builder1.BufPointer()));
  auto size1 = builder1.BufSize();
  auto buf1 = act::BytesBuffer(data1, size1, seastar::make_object_deleter(std::move(builder1)));
  records.emplace_back(std::move(buf1));

  io::RecordBuilder builder2;
  std::string attr2 = "attr2";
  builder2.AddAttribute(0, AttributeValueType::STRING,
                       reinterpret_cast<const int8_t*>(attr2.data()), attr2.size());
  builder2.BuildAsVertexRecord(1, 11);
  auto *data2 = const_cast<char*>(reinterpret_cast<const char*>(builder2.BufPointer()));
  auto size2 = builder2.BufSize();
  auto buf2 = act::BytesBuffer(data2, size2, seastar::make_object_deleter(std::move(builder2)));
  records.emplace_back(std::move(buf2));

  res_builder.Put(0, 11, records);

  res_builder.Finish();
  auto* data = const_cast<char*>(
      reinterpret_cast<const char*>(res_builder.BufPointer()));
  auto size = res_builder.BufSize();
  auto buf = act::BytesBuffer(data, size,
      seastar::make_object_deleter(std::move(res_builder)));
  QueryResponse response(std::move(buf));

  act::SerializableQueue qu;
  response.dump_to(qu);

  auto res = QueryResponse::load_from(qu);
  PrintQueryResponse(res);
}