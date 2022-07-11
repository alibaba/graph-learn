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
  explicit SampleUpdateBatch(actor::BytesBuffer&& buf);

  SampleUpdateBatch(const SampleUpdateBatch&) = delete;
  SampleUpdateBatch& operator=(const SampleUpdateBatch&) = delete;
  SampleUpdateBatch(SampleUpdateBatch&& SampleUpdateBatch) noexcept;
  SampleUpdateBatch& operator=(SampleUpdateBatch&& other) noexcept;

  SampleUpdateBatch(PartitionId pid,
                    const std::vector<const storage::KVPair*>& updates);

  std::vector<storage::KVPair> GetSampleUpdates();

  /// Get the destination partition id of this batch
  PartitionId GetStorePartitionId() const {
    return *reinterpret_cast<const PartitionId*>(buf_.get());
  }

  size_t GetUpdatesNum() const {
    return *reinterpret_cast<const uint32_t*>(buf_.get() + sizeof(PartitionId));
  }

  /// Get the raw data pointer of the underlying buffer.
  const char* Data() const {
    return buf_.get();
  }

  /// Get the raw data size of the underlying buffer.
  size_t Size() const {
    return buf_.size();
  }

  /// Make a new record referring to the same underlying buffer.
  SampleUpdateBatch Share() {
    return SampleUpdateBatch{buf_.share()};
  }

  /// Make a new clone of current record.
  SampleUpdateBatch Clone() const {
    return SampleUpdateBatch{buf_.clone()};
  }

  /// Get the underlying temporary buffer.
  const actor::BytesBuffer& Buffer() const {
    return buf_;
  }

  /// Release the underlying string buffer of this record.
  ///
  /// \remark After calling this method, the current record
  /// is no longer valid.
  actor::BytesBuffer ReleaseBuffer() {
    return std::move(buf_);
  }

  /// As each SampleUpdateBatch polled from kafka will be
  /// ingested locally, the \dump_to and \load_from methods
  /// are implemented with null.
  void dump_to(actor::SerializableQueue& qu) {}  // NOLINT
  static SampleUpdateBatch load_from(actor::SerializableQueue& qu) {  // NOLINT
    return SampleUpdateBatch{};
  }

private:
  actor::BytesBuffer buf_;
};

}  // namespace io
}  // namespace dgs

#endif  // DGS_CORE_IO_SAMPLE_UPDATE_BATCH_H_
