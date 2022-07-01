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

#include "core/io/sample_update_batch.h"

namespace dgs {
namespace io {

SampleUpdateBatch::SampleUpdateBatch(actor::BytesBuffer&& buf)
  : buf_(std::move(buf)) {}

// Layout of buffer:
// partition_id | records_num | key_1 | record_size_1 | record_1, ...
SampleUpdateBatch::SampleUpdateBatch(PartitionId pid,
      const std::vector<const storage::KVPair*>& updates) {
  uint32_t buf_size = sizeof(PartitionId);
  uint32_t records_num = updates.size();
  buf_size += sizeof(uint32_t);
  for (int i = 0; i < records_num; i++) {
    buf_size += (sizeof(storage::Key) + sizeof(uint32_t) +
      updates[i]->value.Size());
  }
  // Copy the pid and #records into the buffer, followed by the size
  // and data of each record.
  actor::BytesBuffer buffer(buf_size);
  auto offset = buffer.get_write();
  std::memcpy(offset, &pid, sizeof(PartitionId));
  offset += sizeof(PartitionId);
  std::memcpy(offset, &records_num, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  for (int i = 0; i < records_num; i++) {
    auto& kv_pair = updates[i];
    std::memcpy(offset, &kv_pair->key, sizeof(storage::Key));
    offset += sizeof(storage::Key);
    uint32_t record_size = kv_pair->value.Size();
    std::memcpy(offset, &record_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(offset, kv_pair->value.Data(), record_size);
    offset += record_size;
  }
  buf_ = std::move(buffer);
}

std::vector<storage::KVPair> SampleUpdateBatch::GetSampleUpdates() {
  std::vector<storage::KVPair> output;
  auto updates_num = GetUpdatesNum();
  auto offset = buf_.get() + sizeof(PartitionId) + sizeof(uint32_t);
  for (int i = 0; i < updates_num; i++) {
    storage::Key key(0, 0, 0, 0);
    std::memcpy(&key, offset, sizeof(storage::Key));
    offset += sizeof(storage::Key);
    uint32_t record_size = *reinterpret_cast<const uint32_t*>(offset);
    offset += sizeof(uint32_t);
    actor::BytesBuffer record_buf = buf_.share(
        offset - buf_.get(), record_size);
    offset += record_size;
    output.emplace_back(storage::KVPair(key, Record({std::move(record_buf)})));
  }
  return output;
}

SampleUpdateBatch::SampleUpdateBatch(SampleUpdateBatch&& other) noexcept
  : buf_(std::move(other.buf_)) {}

SampleUpdateBatch& SampleUpdateBatch::operator=(
    SampleUpdateBatch&& other) noexcept {
  if (this != &other) {
    buf_ = std::move(other.buf_);
  }
  return *this;
}

}  // namespace io
}  // namespace dgs
