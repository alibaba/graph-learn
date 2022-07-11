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

#ifndef DGS_CORE_IO_SUBSCRIPTION_RULE_H_
#define DGS_CORE_IO_SUBSCRIPTION_RULE_H_

#include "common/actor_wrapper.h"
#include "common/typedefs.h"
#include "common/slice.h"
#include "core/storage/key.h"

namespace dgs {
namespace io {

struct SubsRule {
  using Prefix = storage::Key::Prefix;

  SubsRule(VertexType tp, VertexId vid, OperatorId op_id, WorkerId wid)
    : pkey(tp, vid, op_id), worker_id(wid) {}
  explicit SubsRule(const SubsRule* other)
    : pkey(other->pkey), worker_id(other->worker_id) {}

  Slice ToSlice() const {
    return {reinterpret_cast<const char*>(this), sizeof(SubsRule)};
  }

  static SubsRule FromSlice(const Slice& slice) {
    assert(slice.size() == sizeof(SubsRule));
    return SubsRule{reinterpret_cast<const SubsRule*>(slice.data())};
  }

  Prefix   pkey;
  uint64_t worker_id;
};

/// In order to avoid undefined behavior cause by structure padding,
/// which will not guarantee the uniqueness of directly casting
/// `struct SubsRule` to raw byte type(char*).
static_assert(alignof(SubsRule::pkey) == alignof(SubsRule::worker_id));

struct SubsRuleBatch {
  SubsRuleBatch() = default;
  SubsRuleBatch(const SubsRule* begin, const SubsRule* end)
    : rules(begin, end) {}

  SubsRuleBatch(SubsRuleBatch&& other) = default;
  SubsRuleBatch& operator=(SubsRuleBatch&& other) = default;

  void dump_to(actor::SerializableQueue &qu) {  // NOLINT
    actor::BytesBuffer buf(sizeof(uint32_t) + rules.size() * sizeof(SubsRule));
    auto *ptr = buf.get_write();
    uint32_t size = rules.size();
    std::memcpy(ptr, &size, sizeof(uint32_t));
    std::memcpy(ptr + sizeof(uint32_t), rules.data(),
                rules.size() * sizeof(SubsRule));
    qu.push(std::move(buf));
  }

  static SubsRuleBatch load_from(actor::SerializableQueue &qu) {  // NOLINT
    auto buf = qu.pop();
    auto *ptr = buf.get_write();
    uint32_t size = *reinterpret_cast<uint32_t*>(ptr);
    auto *begin = reinterpret_cast<SubsRule*>(ptr + sizeof(uint32_t));
    return SubsRuleBatch(begin, begin + size);
  }

  std::vector<SubsRule> rules;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_SUBSCRIPTION_RULE_H_
