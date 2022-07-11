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

bool AttributeView::AsBool() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::BOOL);
  assert(rep_->value_bytes()->size() == sizeof(bool));
  return *reinterpret_cast<const bool*>(rep_->value_bytes()->data());
}

int8_t AttributeView::AsChar() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::CHAR);
  assert(rep_->value_bytes()->size() == sizeof(int8_t));
  return *reinterpret_cast<const int8_t*>(rep_->value_bytes()->data());
}

int16_t AttributeView::AsInt16() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::INT16);
  assert(rep_->value_bytes()->size() == sizeof(int16_t));
  return *reinterpret_cast<const int16_t*>(rep_->value_bytes()->data());
}

int32_t AttributeView::AsInt32() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::INT32);
  assert(rep_->value_bytes()->size() == sizeof(int32_t));
  return *reinterpret_cast<const int32_t*>(rep_->value_bytes()->data());
}

int64_t AttributeView::AsInt64() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::INT64);
  assert(rep_->value_bytes()->size() == sizeof(int64_t));
  return *reinterpret_cast<const int64_t*>(rep_->value_bytes()->data());
}

float AttributeView::AsFloat32() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::FLOAT32);
  assert(rep_->value_bytes()->size() == sizeof(float));
  return *reinterpret_cast<const float*>(rep_->value_bytes()->data());
}

double AttributeView::AsFloat64() const {
  assert(Valid());
  assert(ValueType() == AttributeValueType::FLOAT64);
  assert(rep_->value_bytes()->size() == sizeof(double));
  return *reinterpret_cast<const double*>(rep_->value_bytes()->data());
}

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
