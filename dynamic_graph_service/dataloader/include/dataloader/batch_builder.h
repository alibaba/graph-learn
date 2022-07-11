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

#ifndef DATALOADER_BATCH_BUILDER_H_
#define DATALOADER_BATCH_BUILDER_H_

#include "dataloader/typedefs.h"
#include "dataloader/fbs/record_generated.h"

namespace dgs {
namespace dataloader {

struct AttrInfo {
  AttributeType attr_type = 0;
  AttributeValueType value_type = INT32;
  std::string value_bytes;

  AttrInfo() = default;
  AttrInfo(AttributeType attr_type, AttributeValueType value_type, std::string&& value_bytes)
    : attr_type(attr_type), value_type(value_type), value_bytes(std::move(value_bytes)) {}
};

/// The graph update record batch builder based on flatbuffers.
/// Use this to serialize the incoming graph update records into output flatbuffers format.
class BatchBuilder {
public:
  explicit BatchBuilder(PartitionId data_partition): partition_id_(data_partition), builder_(1024), records_() {}
  ~BatchBuilder() = default;

  BatchBuilder(const BatchBuilder&) = delete;
  BatchBuilder& operator=(const BatchBuilder&) = delete;
  BatchBuilder(BatchBuilder&&) = default;
  BatchBuilder& operator=(BatchBuilder&&) = default;

  /// Add vertex/edge update record from all fields into builder.
  void AddVertexUpdate(VertexType vtype, VertexId vid, const std::vector<AttrInfo>& attrs);
  void AddEdgeUpdate(EdgeType etype, VertexType src_vtype, VertexType dst_vtype,
                     VertexId src_vid, VertexId dst_vid, const std::vector<AttrInfo>& attrs);

  /// Get the serialized buffer pointer and size.
  const uint8_t* GetBufferPointer() const;
  uint32_t GetBufferSize() const;

  /// Get the current record number in this batch builder
  size_t RecordNum() const {
    return records_.size();
  }

  /// Get the data partition id of this batch builder
  PartitionId GetPartitionId() const {
    return partition_id_;
  }

  /// Finish the building for current batch
  void Finish();

  /// Clear the buffer states
  void Clear();

protected:
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
  AddAttributes(const std::vector<AttrInfo>& attrs);

protected:
  PartitionId partition_id_;
  flatbuffers::FlatBufferBuilder builder_;
  std::vector<flatbuffers::Offset<RecordRep>> records_;
};

inline
void BatchBuilder::AddVertexUpdate(VertexType vtype, VertexId vid, const std::vector<AttrInfo>& attrs) {
  auto flat_attrs = AddAttributes(attrs);
  auto flat_vertex = CreateVertexRecordRep(
      builder_, vtype, vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_VertexRecordRep, flat_vertex.Union());
  records_.emplace_back(flat_record);
}

inline
void BatchBuilder::AddEdgeUpdate(EdgeType etype, VertexType src_vtype, VertexType dst_vtype,
                                 VertexId src_vid, VertexId dst_vid, const std::vector<AttrInfo>& attrs) {
  auto flat_attrs = AddAttributes(attrs);
  auto flat_edge = CreateEdgeRecordRep(
      builder_, etype, src_vtype, dst_vtype, src_vid, dst_vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_EdgeRecordRep, flat_edge.Union());
  records_.emplace_back(flat_record);
}

inline
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

inline
const uint8_t* BatchBuilder::GetBufferPointer() const {
  return builder_.GetBufferPointer();
}

inline
uint32_t BatchBuilder::GetBufferSize() const {
  return builder_.GetSize();
}

inline
void BatchBuilder::Finish() {
  auto flat_records = builder_.CreateVector(records_);
  auto batch = CreateRecordBatchRep(builder_, flat_records, partition_id_);
  builder_.Finish(batch);
}

inline
void BatchBuilder::Clear() {
  builder_.Clear();
  records_.clear();
}

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_BATCH_BUILDER_H_
