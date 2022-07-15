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

#include "dataloader/batch_builder.h"

namespace dgs {
namespace dataloader {

void BatchBuilder::AddVertexUpdate(VertexType vtype, VertexId vid, const std::vector<AttrInfo>& attrs) {
  auto flat_attrs = AddAttributes(attrs);
  auto flat_vertex = CreateVertexRecordRep(
      builder_, vtype, vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_VertexRecordRep, flat_vertex.Union());
  records_.emplace_back(flat_record);
}

void BatchBuilder::AddEdgeUpdate(EdgeType etype, VertexType src_vtype, VertexType dst_vtype,
                                 VertexId src_vid, VertexId dst_vid, const std::vector<AttrInfo>& attrs) {
  auto flat_attrs = AddAttributes(attrs);
  auto flat_edge = CreateEdgeRecordRep(
      builder_, etype, src_vtype, dst_vtype, src_vid, dst_vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_EdgeRecordRep, flat_edge.Union());
  records_.emplace_back(flat_record);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
BatchBuilder::AddAttributes(const std::vector<AttrInfo>& attrs) {
  std::vector<flatbuffers::Offset<AttributeRecordRep>> flat_attrs_vec;
  flat_attrs_vec.reserve(attrs.size());
  for (auto& a: attrs) {
    auto flat_value_type = static_cast<AttributeValueTypeRep>(a.value_type);
    auto flat_value_bytes = builder_.CreateVector(
        reinterpret_cast<const int8_t*>(a.value_bytes.data()), a.value_bytes.size());
    auto flat_attr = CreateAttributeRecordRep(
        builder_, a.attr_type, flat_value_type, flat_value_bytes);
    flat_attrs_vec.push_back(flat_attr);
  }
  return builder_.CreateVectorOfSortedTables(&flat_attrs_vec);
}

const uint8_t* BatchBuilder::GetBufferPointer() const {
  return builder_.GetBufferPointer();
}

uint32_t BatchBuilder::GetBufferSize() const {
  return builder_.GetSize();
}

void BatchBuilder::Finish() {
  auto flat_records = builder_.CreateVector(records_);
  auto batch = CreateRecordBatchRep(builder_, flat_records, partition_id_);
  builder_.Finish(batch);
}

void BatchBuilder::Clear() {
  builder_.Clear();
  records_.clear();
}

}  // namespace dataloader
}  // namespace dgs

