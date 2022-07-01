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

#ifndef DGS_CORE_STORAGE_SAMPLER_H_
#define DGS_CORE_STORAGE_SAMPLER_H_

#include <vector>

#include "common/actor_wrapper.h"
#include "common/schema.h"
#include "core/io/record.h"

namespace dgs {
namespace storage {

template <typename RecordType>
class Sampler {
public:
  Sampler() = default;
  virtual ~Sampler() {}

  virtual bool Sample(const RecordType& sample,
                      uint32_t& index) = 0;  // NOLINT

  virtual actor::BytesBuffer Dump() = 0;
  virtual void Load(const actor::BytesBuffer& slice) = 0;
};

class VertexSampler : public Sampler<io::VertexRecordView> {
public:
  struct Entry {
    Timestamp timestamp;
    uint32_t  index;
  };

  struct Comparator {
    bool operator() (const Entry& a, const Entry& b) {
      return a.timestamp > b.timestamp;
    }
  };

  explicit VertexSampler(Capacity cap)
    : Sampler<io::VertexRecordView>(),
      timestamp_type_(
        Schema::GetInstance().GetAttrDefByName("timestamp").Type()),
      capacity_(cap) {
    assert(capacity_ > 0);
    samples_.reserve(capacity_);
  }

  ~VertexSampler() override = default;

  bool Sample(const io::VertexRecordView& sample, uint32_t& index) override;

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

#endif  // DGS_CORE_STORAGE_SAMPLER_H_
