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

#ifndef DGS_CORE_STORAGE_TOPK_SAMPLER_H_
#define DGS_CORE_STORAGE_TOPK_SAMPLER_H_

#include "core/io/record.h"
#include "core/storage/sample_builder.h"

namespace dgs {
namespace storage {

using EdgeSampler = Sampler<io::EdgeRecordView>;

class TopKSampler : public EdgeSampler {
  struct Entry {
    Timestamp timestamp;
    uint32_t  index;
  };

  struct Comparator {
    bool operator() (const Entry& a, const Entry& b) {
      return a.timestamp > b.timestamp;
    }
  };

public:
  explicit TopKSampler(Capacity cap)
    : EdgeSampler(),
      timestamp_type_(
        Schema::GetInstance().GetAttrDefByName("timestamp").Type()),
      capacity_(cap) {}
  ~TopKSampler() override = default;

  bool Sample(const io::EdgeRecordView& sample, uint32_t& index) override;

  actor::BytesBuffer Dump() override;
  void Load(const actor::BytesBuffer& buffer) override;

private:
  AttributeType       timestamp_type_;
  Capacity            capacity_;
  static Comparator   comparator_;
  std::vector<Entry>  samples_;
};


}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_TOPK_SAMPLER_H_
