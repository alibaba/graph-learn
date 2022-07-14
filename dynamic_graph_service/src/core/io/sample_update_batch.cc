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

SampleUpdateBatch::SampleUpdateBatch(PartitionId store_pid,
                                     std::vector<storage::KVPair>&& updates)
  : store_pid_(store_pid), updates_(std::move(updates)) {
}

actor::BytesBuffer
SampleUpdateBatch::Serialize(const storage::KVPair* const* updates, uint32_t size) {
  uint32_t buf_size = 0;
  buf_size += sizeof(uint32_t);
  for (uint32_t i = 0; i < size; i++) {
    buf_size += (sizeof(storage::Key) + sizeof(uint32_t) +
        updates[i]->value.Size());
  }
  // Copy the pid and #records into the buffer, followed by the size
  // and data of each record.
  actor::BytesBuffer buffer(buf_size);
  auto offset = buffer.get_write();
  // write record number
  std::memcpy(offset, &size, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  // write updates
  for (uint32_t i = 0; i < size; i++) {
    auto* kv_pair = updates[i];
    std::memcpy(offset, &kv_pair->key, sizeof(storage::Key));
    offset += sizeof(storage::Key);
    uint32_t update_size = kv_pair->value.Size();
    std::memcpy(offset, &update_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(offset, kv_pair->value.Data(), update_size);
    offset += update_size;
  }
  return buffer;
}

std::vector<storage::KVPair>
SampleUpdateBatch::Deserialize(actor::BytesBuffer&& buf) {
  auto updates_num = *reinterpret_cast<const uint32_t*>(buf.get());
  std::vector<storage::KVPair> output;
  output.reserve(updates_num);
  auto offset = buf.get() + sizeof(uint32_t);
  for (int i = 0; i < updates_num; i++) {
    storage::Key key(0, 0, 0, 0);
    std::memcpy(&key, offset, sizeof(storage::Key));
    offset += sizeof(storage::Key);
    uint32_t update_size = *reinterpret_cast<const uint32_t*>(offset);
    offset += sizeof(uint32_t);
    actor::BytesBuffer update_buf = buf.share(offset - buf.get(), update_size);
    offset += update_size;
    output.emplace_back(key, Record({std::move(update_buf)}));
  }
  return output;
}

}  // namespace io
}  // namespace dgs
