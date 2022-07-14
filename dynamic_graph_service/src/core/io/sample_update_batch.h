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

#ifndef DGS_CORE_IO_SAMPLE_UPDATE_BATCH_H_
#define DGS_CORE_IO_SAMPLE_UPDATE_BATCH_H_

#include "common/actor_wrapper.h"
#include "core/storage/sample_store.h"

namespace dgs {
namespace io {

class SampleUpdateBatch {
public:
  SampleUpdateBatch() = default;
  SampleUpdateBatch(PartitionId store_pid,
                    std::vector<storage::KVPair>&& updates);

  SampleUpdateBatch(const SampleUpdateBatch&) = delete;
  SampleUpdateBatch& operator=(const SampleUpdateBatch&) = delete;
  SampleUpdateBatch(SampleUpdateBatch&&) noexcept = default;
  SampleUpdateBatch& operator=(SampleUpdateBatch&&) noexcept = default;

  /// Serialize updates without store partition info
  static actor::BytesBuffer
  Serialize(const storage::KVPair** updates, uint32_t size);

  /// Deserialize updates from buffer without store partition info
  static std::vector<storage::KVPair>
  Deserialize(actor::BytesBuffer&& buf);

  /// Get the destination store partition id of this batch
  PartitionId GetStorePartitionId() const {
    return store_pid_;
  }

  size_t GetUpdatesNum() const {
    return updates_.size();
  }

  std::vector<storage::KVPair> ReleaseUpdates() {
    return std::move(updates_);
  }

  /// As each SampleUpdateBatch polled from kafka will be
  /// ingested locally, the \dump_to and \load_from methods
  /// are implemented with null.
  void dump_to(actor::SerializableQueue& qu) {}  // NOLINT
  static SampleUpdateBatch load_from(actor::SerializableQueue& qu) {  // NOLINT
    return SampleUpdateBatch{};
  }

private:
  PartitionId store_pid_ = 0;
  std::vector<storage::KVPair> updates_;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_SAMPLE_UPDATE_BATCH_H_
