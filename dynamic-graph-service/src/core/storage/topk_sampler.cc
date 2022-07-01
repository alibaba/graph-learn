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

#include "core/storage/topk_sampler.h"

#include "core/storage/sampler_factory.h"

namespace dgs {
namespace storage {

bool TopKSampler::Sample(const io::EdgeRecordView& sample, uint32_t& index) {
  Timestamp timestamp = sample.LookUpAttrByType(timestamp_type_).AsInt64();
  if (samples_.size() < capacity_) {
    index = samples_.size();
    samples_.emplace_back(Entry{timestamp, index});
    return true;
  } else {
    std::make_heap(samples_.begin(), samples_.end(), comparator_);
    pop_heap(samples_.begin(), samples_.end(), comparator_);
    auto eldest = samples_.back();
    if (timestamp > eldest.timestamp) {
      index = eldest.index;
      samples_.pop_back();
      samples_.emplace_back(Entry{timestamp, index});
      return true;
    }
    return false;
  }
}

actor::BytesBuffer TopKSampler::Dump() {
  size_t size = sizeof(Entry) * samples_.size();
  actor::BytesBuffer buffer(size);
  auto data = buffer.get_write();
  std::memcpy(data, &samples_[0], size);
  return std::move(buffer);
}

void TopKSampler::Load(const actor::BytesBuffer& buffer) {
  auto data = buffer.get();
  size_t entry_num = buffer.size() / sizeof(Entry);
  assert(samples_.size() == 0);
  samples_.resize(entry_num);
  std::memcpy(&samples_[0], data, sizeof(Entry) * entry_num);
}

// Register in sampler factory
registration::EdgeSamplerRegistration<TopKSampler> _topk_edge_sampler(
  SamplerType::TOPK_BY_TIMESTAMP);

}  // namespace storage
}  // namespace dgs
