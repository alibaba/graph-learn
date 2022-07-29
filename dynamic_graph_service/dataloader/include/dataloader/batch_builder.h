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

#include "dataloader/attribute.h"
#include "dataloader/typedefs.h"
#include "dataloader/fbs/record_generated.h"

namespace dgs {
namespace dataloader {

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

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_BATCH_BUILDER_H_
