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

#ifndef DGS_CORE_IO_RECORD_VIEW_H_
#define DGS_CORE_IO_RECORD_VIEW_H_

#include <string>
#include <vector>

#include "common/typedefs.h"
#include "generated/fbs/record_generated.h"

namespace dgs {
namespace io {

/// Use this structure to view an io attribute record.
///
/// During the use of an \AttributeView, the rep pointer
/// held by it needs to remain valid.
class AttributeView {
public:
  AttributeView() : rep_(nullptr) {}
  explicit AttributeView(const AttributeRecordRep* rep) : rep_(rep) {}

  bool Valid() const {
    return rep_ != nullptr;
  }

  AttributeType AttrType() const {
    assert(Valid());
    return rep_->attr_type();
  }

  AttributeValueType ValueType() const {
    assert(Valid());
    return static_cast<AttributeValueType>(rep_->value_type());
  }

  const int8_t* ValueBytesData() const {
    assert(Valid());
    return rep_->value_bytes()->data();
  }

  size_t ValueBytesSize() const {
    assert(Valid());
    return rep_->value_bytes()->size();
  }

  const AttributeRecordRep* RepPointer() const {
    return rep_;
  }

  bool AsBool() const;
  int8_t AsChar() const;
  int16_t AsInt16() const;
  int32_t AsInt32() const;
  int64_t AsInt64() const;
  float AsFloat32() const;
  double AsFloat64() const;
  std::string AsString() const;  // with copy

protected:
  const AttributeRecordRep* rep_;
};

/// Use this structure to view an io vertex update record.
///
/// During the use of a \VertexRecordView, the rep pointer
/// held by it needs to remain valid.
class VertexRecordView {
public:
  VertexRecordView() : rep_(nullptr) {}
  explicit VertexRecordView(const VertexRecordRep* rep) : rep_(rep) {}

  bool Valid() const {
    return rep_ != nullptr;
  }

  VertexType Type() const {
    assert(Valid());
    return rep_->vtype();
  }

  VertexId Id() const {
    assert(Valid());
    return rep_->vid();
  }

  size_t AttrNum() const {
    assert(Valid());
    return rep_->attributes()->size();
  }

  AttributeView GetAttrByIdx(size_t idx) const {
    assert(Valid());
    return AttributeView{rep_->attributes()->Get(idx)};
  }

  /// Look up the record attribute by its attribute type directly.
  AttributeView LookUpAttrByType(AttributeType type) const {
    assert(Valid());
    return AttributeView{rep_->attributes()->LookupByKey(type)};
  }

  std::vector<AttributeView> GetAllAttrs() const;

  const VertexRecordRep* RepPointer() const {
    return rep_;
  }

protected:
  const VertexRecordRep* rep_;
};

/// Use this structure to view an io edge record.
///
/// During the use of an \EdgeRecordView, the rep pointer
/// held by it needs to remain valid.
class EdgeRecordView {
public:
  EdgeRecordView() : rep_(nullptr) {}
  explicit EdgeRecordView(const EdgeRecordRep* rep) : rep_(rep) {}

  bool Valid() const {
    return rep_ != nullptr;
  }

  EdgeType Type() const {
    assert(Valid());
    return rep_->etype();
  }

  VertexType SrcType() const {
    assert(Valid());
    return rep_->src_vtype();
  }

  VertexType DstType() const {
    assert(Valid());
    return rep_->dst_vtype();
  }

  VertexId SrcId() const {
    assert(Valid());
    return rep_->src_id();
  }

  VertexId DstId() const {
    assert(Valid());
    return rep_->dst_id();
  }

  size_t AttrNum() const {
    assert(Valid());
    return rep_->attributes()->size();
  }

  AttributeView GetAttrByIdx(size_t idx) const {
    assert(Valid());
    return AttributeView{rep_->attributes()->Get(idx)};
  }

  AttributeView LookUpAttrByType(AttributeType type) const {
    assert(Valid());
    return AttributeView{rep_->attributes()->LookupByKey(type)};
  }

  std::vector<AttributeView> GetAllAttrs() const;

  const EdgeRecordRep* RepPointer() const {
    return rep_;
  }

protected:
  const EdgeRecordRep* rep_;
};

/// Use this structure to view an io record.
///
/// During the use of a \RecordView, the rep pointer
/// held by it needs to remain valid.
class RecordView {
public:
  RecordView() : rep_(nullptr) {}
  explicit RecordView(const RecordRep* rep) : rep_(rep) {}

  bool Valid() const {
    return rep_ != nullptr;
  }

  RecordType Type() const {
    assert(Valid());
    return static_cast<RecordType>(rep_->record_type());
  }

  VertexRecordView AsVertexRecord() const {
    assert(Valid());
    assert(Type() == RecordType::VERTEX);
    return VertexRecordView{rep_->record_as_VertexRecordRep()};
  }

  EdgeRecordView AsEdgeRecord() const {
    assert(Valid());
    assert(Type() == RecordType::EDGE);
    return EdgeRecordView{rep_->record_as_EdgeRecordRep()};
  }

  const RecordRep* RepPointer() const {
    return rep_;
  }

protected:
  const RecordRep* rep_;
};

/// Use this structure to view an io record batch.
///
/// During the use of a \RecordBatchView, the rep pointer
/// held by it needs to remain valid.
class RecordBatchView {
public:
  RecordBatchView() : rep_(nullptr) {}
  explicit RecordBatchView(const RecordBatchRep* rep) : rep_(rep) {}

  bool Valid() const {
    return rep_ != nullptr;
  }

  PartitionId GetStorePartitionId() const {
    assert(Valid());
    return rep_->partition();
  }

  size_t RecordNum() const {
    assert(Valid());
    return rep_->records()->size();
  }

  RecordView GetRecordByIdx(size_t idx) const {
    assert(Valid());
    return RecordView{rep_->records()->Get(idx)};
  }

  std::vector<RecordView> GetAllRecords() const;

  const RecordBatchRep* RepPointer() const {
    return rep_;
  }

protected:
  const RecordBatchRep* rep_;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_RECORD_VIEW_H_
