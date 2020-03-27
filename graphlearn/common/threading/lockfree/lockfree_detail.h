/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_COMMON_THREADING_LOCKFREE_LOCKFREE_DETAIL_H_
#define GRAPHLEARN_COMMON_THREADING_LOCKFREE_LOCKFREE_DETAIL_H_

#include <cstddef>
#include <cstdint>

namespace graphlearn {
namespace detail {

const uint32_t kLockFreeNullPointer = 0xFFFFFFFF;

using Pointer = uint64_t;

inline uint64_t MakePointer(uint32_t index, uint32_t tag) {
  uint64_t value = index;
  value <<= 32;
  value += tag;
  return value;
}

inline uint32_t Tag(uint64_t point) {
  return point & 0xFFFFFFFF;
}

inline uint32_t Index(uint64_t point) {
  return point >> 32;
}

union TaggedPointer {
  uint64_t mValue;
  void* mPointer;
  struct {
    uint16_t mDummy[3];  // padding
    uint16_t mTag;
  };
};

const uint64_t kX64AddressSignBit = 0x0000800000000000;
const uint16_t kInvalidTag = 0xdead;

inline uint64_t MakePointer(void* address, uint16_t tag) {
  TaggedPointer pointer;
  pointer.mValue = static_cast<uint64_t>(reinterpret_cast<intptr_t>(address));
  pointer.mTag = tag;
  return pointer.mValue;
}

inline void* GetPointer(uint64_t pointer) {
  TaggedPointer tagged_pointer;
  tagged_pointer.mValue = pointer;
  tagged_pointer.mTag = (pointer & kX64AddressSignBit) ? 0xFFFF : 0;
  return  reinterpret_cast<void*>(static_cast<intptr_t>(tagged_pointer.mValue));
}

inline uint16_t GetTag(uint64_t pointer) {
  TaggedPointer tagged_pointer;
  tagged_pointer.mValue = pointer;
  return tagged_pointer.mTag;
}

inline uint16_t GetNextTag(uint64_t pointer) {
  TaggedPointer tagged_pointer;
  tagged_pointer.mValue = pointer;
  uint16_t tag = tagged_pointer.mTag + 1;
  if (__builtin_expect(tag == kInvalidTag, 0)) {
    ++tag;
  }
  return tag;
}

inline uint16_t GetPrevTag(uint64_t pointer) {
  TaggedPointer tagged_pointer;
  tagged_pointer.mValue = pointer;
  uint16_t tag = tagged_pointer.mTag - 1;
  if (__builtin_expect(tag == kInvalidTag, 0)) {
    --tag;
  }
  return tag;
}

}  // namespace detail
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_LOCKFREE_LOCKFREE_DETAIL_H_
