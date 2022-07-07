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

#include "core/io/record_builder.h"

#include "common/typedefs.h"
#include "core/io/record_view.h"

namespace dgs {
namespace io {

RecordBuilderBase::RecordOffset RecordBuilderBase::WriteVertex(
  VertexType type, VertexId id, RecordBuilderBase::AttrVecOffset attrs) {
  auto flat_vertex = CreateVertexRecordRep(builder_, type, id, attrs);
  return CreateRecordRep(
    builder_, RecordUnionRep_VertexRecordRep, flat_vertex.Union());
}

RecordBuilderBase::RecordOffset RecordBuilderBase::WriteEdge(
  EdgeType etype, VertexType src_type, VertexType dst_type,
  VertexId src_id, VertexId dst_id, RecordBuilderBase::AttrVecOffset attrs) {
  auto flat_edge = CreateEdgeRecordRep(
    builder_, etype, src_type, dst_type, src_id, dst_id, attrs);
  return CreateRecordRep(
    builder_, RecordUnionRep_EdgeRecordRep, flat_edge.Union());
}

RecordBuilderBase::AttrOffset RecordBuilderBase::WriteAttribute(
    AttributeType type, AttributeValueType value_type,
    const int8_t* value_bytes_ptr, size_t value_bytes_size) {
  auto flat_value_bytes = builder_.CreateVector(
    value_bytes_ptr, value_bytes_size);
  return CreateAttributeRecordRep(builder_,
    type, static_cast<AttributeValueTypeRep>(value_type), flat_value_bytes);
}

RecordBuilderBase::AttrOffset RecordBuilderBase::WriteAttribute(
    const AttributeView& attr) {
  return WriteAttribute(attr.AttrType(), attr.ValueType(),
                        attr.ValueBytesData(), attr.ValueBytesSize());
}

RecordBuilderBase::AttrVecOffset RecordBuilderBase::WriteAllAttributes(
    const std::vector<AttributeView>& attrs) {
  std::vector<flatbuffers::Offset<AttributeRecordRep>> flat_attrs_vec;
  flat_attrs_vec.reserve(attrs.size());
  for (auto& attr : attrs) {
    flat_attrs_vec.emplace_back(WriteAttribute(attr));
  }
  return builder_.CreateVectorOfSortedTables(&flat_attrs_vec);
}

void RecordBuilder::AddAttributeBatch(const std::vector<AttributeView>& attrs) {
  for (auto& a : attrs) {
    attr_vec_.emplace_back(WriteAttribute(a));
  }
}

void RecordBuilder::BuildAsVertexRecord(VertexType type, VertexId id) {
  builder_.Finish(WriteVertex(type, id, builder_.CreateVector(attr_vec_)));
}

void RecordBuilder::BuildAsEdgeRecord(EdgeType etype, VertexType src_type,
                                      VertexType dst_type, VertexId src_id,
                                      VertexId dst_id) {
  builder_.Finish(WriteEdge(
    etype, src_type, dst_type, src_id, dst_id,
    builder_.CreateVector(attr_vec_)));
}

void RecordBuilder::BuildFromView(const VertexRecordView* view) {
  Clear();
  auto flat_attrs = WriteAllAttributes(view->GetAllAttrs());
  builder_.Finish(WriteVertex(view->Type(), view->Id(), flat_attrs));
}

void RecordBuilder::BuildFromView(const EdgeRecordView* view) {
  Clear();
  auto flat_attrs = WriteAllAttributes(view->GetAllAttrs());
  builder_.Finish(WriteEdge(
    view->Type(), view->SrcType(), view->DstType(),
    view->SrcId(), view->DstId(), flat_attrs));
}

void RecordBuilder::Clear() {
  builder_.Clear();
  attr_vec_.clear();
}

void RecordBatchBuilder::AddRecord(const RecordView& view) {
  switch (view.Type()) {
    case RecordType::VERTEX: {
      auto vv = view.AsVertexRecord();
      record_vec_.emplace_back(WriteVertex(
        vv.Type(), vv.Id(), WriteAllAttributes(vv.GetAllAttrs())));
      break;
    }
    case RecordType::EDGE: {
      auto ev = view.AsEdgeRecord();
      record_vec_.emplace_back(WriteEdge(
        ev.Type(), ev.SrcType(), ev.DstType(),
        ev.SrcId(), ev.DstId(), WriteAllAttributes(ev.GetAllAttrs())));
      break;
    }
    default: {
      // TODO(@houbai.zzc): log with warning info
    }
  }
}

void RecordBatchBuilder::AddRecord(const RecordBuilder& builder) {
  auto rep = flatbuffers::GetRoot<RecordRep>(builder.BufPointer());
  RecordView view(rep);
  AddRecord(view);
}

void RecordBatchBuilder::Finish() {
  auto flat_records = builder_.CreateVector(record_vec_);
  auto batch = CreateRecordBatchRep(builder_, flat_records, store_pid_);
  builder_.Finish(batch);
}

void RecordBatchBuilder::Clear() {
  builder_.Clear();
  record_vec_.clear();
}

}  // namespace io
}  // namespace dgs
