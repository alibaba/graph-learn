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

#include "core/io/record_view.h"

namespace dgs {
namespace io {

std::string AttributeView::AsString() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::STRING);
  return {reinterpret_cast<const char*>(rep_->value_bytes()->data()),
          rep_->value_bytes()->size()};
}

std::vector<AttributeView> VertexRecordView::GetAllAttrs() const {
  assert(Valid());
  std::vector<AttributeView> attrs;
  attrs.reserve(rep_->attributes()->size());
  for (auto a : *rep_->attributes()) {
    attrs.emplace_back(a);
  }
  return attrs;
}

std::vector<AttributeView> EdgeRecordView::GetAllAttrs() const {
  assert(Valid());
  std::vector<AttributeView> attrs;
  attrs.reserve(rep_->attributes()->size());
  for (auto a : *rep_->attributes()) {
    attrs.emplace_back(a);
  }
  return attrs;
}

std::vector<RecordView> RecordBatchView::GetAllRecords() const {
  assert(Valid());
  std::vector<RecordView> records;
  records.reserve(rep_->records()->size());
  for (auto a : *rep_->records()) {
    records.emplace_back(a);
  }
  return records;
}

}  // namespace io
}  // namespace dgs
