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

#ifndef DGS_CORE_STORAGE_KEY_H_
#define DGS_CORE_STORAGE_KEY_H_

#include <cassert>
#include <type_traits>

#include "common/slice.h"
#include "common/typedefs.h"

namespace dgs {
namespace storage {

struct Key {
  struct Prefix {
    int64_t vid;
    int32_t vtype;
    int32_t op_id;

    Prefix(VertexType vtype, VertexId vid, OperatorId op_id)
      : vtype(vtype), vid(vid), op_id(op_id) {}

    inline Slice ToSlice() const;

    inline bool operator==(const Prefix &p) const;

    static_assert(std::is_same<VertexId, int64_t>::value);
    static_assert(std::is_same<VertexType, int32_t>::value);
    static_assert(std::is_same<OperatorId, int32_t>::value);
  };

  Key(VertexType tp, VertexId vid, OperatorId op_id, uint64_t idx)
    : pkey(tp, vid, op_id), index(idx) {}

  explicit Key(const Key* other)
    : pkey(other->pkey), index(other->index) {}

  inline Slice ToSlice() const;
  static inline Key FromSlice(const Slice& slice);

public:
  Prefix   pkey;
  uint64_t index;

  // for alignment concern.
  static_assert(sizeof(Prefix) == 16);
};

/// In order to avoid undefined behavior caused by structure padding,
/// which will not guarantee the uniqueness of directly casting
/// `struct Key` to raw byte type(char*).
static_assert(alignof(Key::pkey) == alignof(Key::index));

inline
Slice Key::Prefix::ToSlice() const {
  return {reinterpret_cast<const char*>(this), sizeof(Key::Prefix)};
}

inline
bool Key::Prefix::operator==(const Key::Prefix &p) const {
  return vtype == p.vtype && vid == p.vid && op_id == p.op_id;
}

inline
Slice Key::ToSlice() const {
  return {reinterpret_cast<const char*>(this), sizeof(Key)};
}

inline
Key Key::FromSlice(const Slice& slice) {
  assert(slice.size() == sizeof(Key));
  return Key{reinterpret_cast<const Key*>(slice.data())};
}

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_KEY_H_
