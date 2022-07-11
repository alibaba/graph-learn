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

#ifndef DGS_CORE_IO_RECORD_BUILDER_H_
#define DGS_CORE_IO_RECORD_BUILDER_H_

#include "flatbuffers/flatbuffers.h"

#include "core/io/record_slice.h"
#include "generated/fbs/record_generated.h"

namespace dgs {
namespace io {

/// The base class of builders for Record/RecordBatch.
class RecordBuilderBase {
public:
  /// Construct the builder with an initial buffer size.
  ///
  /// \remarks When the buffer capacity is not enough, it
  /// will automatically expand.
  explicit RecordBuilderBase(size_t buf_size) : builder_(buf_size) {}

  /// Get the buffer pointer of the serialized flatbuffers table.
  ///
  /// \remark Should be called after building.
  const uint8_t* BufPointer() const {
    return builder_.GetBufferPointer();
  }

  /// Get the buffer size of the serialized flatbuffers table.
  uint32_t BufSize() const {
    return builder_.GetSize();
  }

  /// Reset all the state in current builder so it can be
  /// reused to construct another records.
  virtual void Clear() {
    builder_.Clear();
  }

protected:
  using RecordOffset = flatbuffers::Offset<RecordRep>;
  using AttrOffset = flatbuffers::Offset<AttributeRecordRep>;
  using AttrVecOffset = flatbuffers::Offset<flatbuffers::Vector<
    flatbuffers::Offset<AttributeRecordRep>>>;

  RecordOffset WriteVertex(VertexType type, VertexId id, AttrVecOffset attrs);
  RecordOffset WriteEdge(EdgeType etype,
                         VertexType src_type, VertexType dst_type,
                         VertexId src_id, VertexId dst_id,
                         AttrVecOffset attrs);
  AttrOffset WriteAttribute(AttributeType type,
                            AttributeValueType value_type,
                            const int8_t* value_bytes_ptr,
                            size_t value_bytes_size);
  AttrOffset WriteAttribute(const AttributeView& attr);
  AttrVecOffset WriteAllAttributes(const std::vector<AttributeView>& attrs);

protected:
  flatbuffers::FlatBufferBuilder builder_;
};

/// Use this class to build a new record which
/// corresponds to a serialized \RecordRep of flatbuffers.
class RecordBuilder : public RecordBuilderBase {
public:
  explicit RecordBuilder(size_t buf_size = 512) : RecordBuilderBase(buf_size) {}

  /// Add an attribute record into builder from all fields.
  void AddAttribute(AttributeType type, AttributeValueType value_type,
                    const int8_t* value_bytes_ptr, size_t value_bytes_size) {
    attr_vec_.emplace_back(WriteAttribute(
      type, value_type, value_bytes_ptr, value_bytes_size));
  }

  /// Add an attribute record into builder from an attribute view.
  void AddAttribute(const AttributeView& attr) {
    attr_vec_.emplace_back(WriteAttribute(attr));
  }

  /// Add a batch of attribute records
  void AddAttributeBatch(const std::vector<AttributeView>& attrs);

  /// Build record as a vertex / an edge.
  ///
  /// \remark Before calling this method, all the attributes of
  /// current building record should have been writen into builder.
  ///
  /// \remark After calling this method, the building is finished,
  /// calling \Clear is required to reset buffer states before
  /// re-building other records.
  void BuildAsVertexRecord(VertexType type, VertexId id);
  void BuildAsEdgeRecord(EdgeType etype, VertexType src_type,
                         VertexType dst_type, VertexId src_id,
                         VertexId dst_id);

  /// Build from a record view directly.
  ///
  /// \remark When calling this method, all the previous contents
  /// in this builder will be cleared first. After calling this
  /// method, the building is finished
  void BuildFromView(const VertexRecordView* view);
  void BuildFromView(const EdgeRecordView* view);

  /// Reset all the state in current builder.
  void Clear() override;

private:
  std::vector<flatbuffers::Offset<AttributeRecordRep>> attr_vec_;
};

/// Use this class to build a record batch which
/// corresponds to a serialized \RecordBatchRep of flatbuffers.
class RecordBatchBuilder : public RecordBuilderBase {
public:
  explicit RecordBatchBuilder(size_t buf_size = 1024)
    : RecordBuilderBase(buf_size) {}

  /// Add a record into batch from a record view
  void AddRecord(const RecordView& view);

  /// Add a record into batch from a built record builder.
  void AddRecord(const RecordBuilder& builder);

  /// Get the record number in current batch builder.
  size_t RecordNum() const {
    return record_vec_.size();
  }

  /// Get the storage partition id.
  PartitionId GetStorePartitionId() const {
    return store_pid_;
  }

  /// Set the data partition id.
  void SetStorePartitionId(PartitionId pid) {
    store_pid_ = pid;
  }

  /// Finish building.
  ///
  /// \remarks After calling this method, the building is finished,
  /// calling \Clear is required to reset buffer states before
  /// re-building other batches.
  virtual void Finish();

  /// Reset all the state in current builder.
  void Clear() override;

protected:
  PartitionId store_pid_ = 0;
  std::vector<flatbuffers::Offset<RecordRep>> record_vec_{};
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_RECORD_BUILDER_H_
