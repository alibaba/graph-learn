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

#ifndef DGS_COMMON_TYPEDEFS_H_
#define DGS_COMMON_TYPEDEFS_H_

#include <cstdint>
#include <type_traits>
#include <string>
#include <unordered_map>

namespace dgs {

using VertexId = int64_t;
using VertexType = int32_t;
using EdgeType = int16_t;
using AttributeType = int16_t;

using ShardId = uint32_t;
using WorkerId = uint32_t;
using QueryId = uint32_t;
using PartitionId = uint32_t;
using EdgeTrackerId = uint32_t;
using GlobalBackupId = uint32_t;
using PartitionBackupId = uint32_t;

// FIXME(@goldenleaves): Currently, enum value of RecordType should be
// always exactly same with the enum value of RecordUnionRep
// which is defined in record_generated.h(fbs generated file).
// However, there is no gurantee to it. Modification in one of
// the enum class may cause bugs that hard to detect. Same issue
// exists in the definition of AttributeValueType. We need
// a `define-once-for-all` enum class instead.

// Be consistent with flatbuffers table  "RecordUnionRep"
enum RecordType : uint8_t {
  VERTEX = 0x1,
  EDGE   = 0x2,
};

// Be consistent with flatbuffers table  "AttributeValueTypeRep"
enum AttributeValueType : uint8_t {
  UNSPECIFIED = 0,
  BOOL = 1,
  CHAR = 2,
  INT16 = 3,
  INT32 = 4,
  INT64 = 5,
  FLOAT32 = 6,
  FLOAT64 = 7,
  STRING = 8,
  BYTES = 9,
};

using Timestamp = uint64_t;
using FrontierType = uint64_t;

using QueryId = uint32_t;
using QueryPriority = uint8_t;
using FieldIndex = uint8_t;

using ActorInstanceIdType = uint32_t;

using OperatorId = int32_t;
using Capacity = uint16_t;
using OpParamType = uint8_t;
using ParamMap = std::unordered_map<std::string, OpParamType>;

}  // namespace dgs

#endif  // DGS_COMMON_TYPEDEFS_H_
