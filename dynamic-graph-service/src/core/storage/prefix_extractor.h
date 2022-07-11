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

#ifndef DGS_CORE_STORAGE_PREFIX_EXTRACTOR_H_
#define DGS_CORE_STORAGE_PREFIX_EXTRACTOR_H_

#include "common/utils.h"
#include "core/storage/key.h"
#include "rocksdb/slice_transform.h"

namespace dgs {
namespace storage {

class RocksdbPrefixExtractor : public rocksdb::SliceTransform {
public:
  ~RocksdbPrefixExtractor() override {}

  const char* Name() const override {
    return "RocksdbPrefixExtractor";
  }

  rocksdb::Slice Transform(const rocksdb::Slice& key) const override {
    return rocksdb::Slice(key.data(), sizeof(Key::Prefix));
  }

  bool InDomain(const rocksdb::Slice& key) const override {
    return key.size() >= sizeof(Key::Prefix);
  }
};

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_PREFIX_EXTRACTOR_H_
