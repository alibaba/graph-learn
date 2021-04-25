/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "graphlearn/core/io/element_value.h"

#include <mutex>  //NOLINT [build/c++11]
#include <unordered_map>
#include "graphlearn/include/config.h"
#include "graphlearn/common/threading/sync/lock.h"

namespace graphlearn {
namespace io {

AttributeValue* AttributeValue::Default(const SideInfo* info) {
  static std::mutex mtx;
  static std::unordered_map<std::string, AttributeValue*> buffer;

  ScopedLocker<std::mutex> _(&mtx);
  auto it = buffer.find(info->type);
  if (it != buffer.end()) {
    return it->second;
  }

  AttributeValue* attr = NewDataHeldAttributeValue();
  attr->Reserve(info->i_num, info->f_num, info->s_num);
  buffer[info->type] = attr;

  for (int32_t i = 0; i < info->i_num; ++i) {
    attr->Add(GLOBAL_FLAG(DefaultIntAttribute));
  }
  for (int32_t i = 0; i < info->f_num; ++i) {
    attr->Add(GLOBAL_FLAG(DefaultFloatAttribute));
  }
  for (int32_t i = 0; i < info->s_num; ++i) {
    attr->Add(GLOBAL_FLAG(DefaultStringAttribute));
  }
  return attr;
}

class DataHeldAttributeValue : public AttributeValue {
public:
  DataHeldAttributeValue() = default;
  DataHeldAttributeValue(const DataHeldAttributeValue& right) {
    i_attrs_ = right.i_attrs_;
    f_attrs_ = right.f_attrs_;
    s_attrs_ = right.s_attrs_;
  }

  void Clear() override {
    i_attrs_.clear();
    f_attrs_.clear();
    s_attrs_.clear();
  }

  void Shrink() override {
    i_attrs_.shrink_to_fit();
    f_attrs_.shrink_to_fit();
    s_attrs_.shrink_to_fit();
  }

  void Swap(AttributeValue* rhs) override {
    DataHeldAttributeValue* right = static_cast<DataHeldAttributeValue*>(rhs);
    i_attrs_.swap(right->i_attrs_);
    f_attrs_.swap(right->f_attrs_);
    s_attrs_.swap(right->s_attrs_);
    s_lites_.swap(right->s_lites_);
  }

  void Reserve(int32_t i_num, int32_t f_num, int32_t s_num) override {
    i_attrs_.reserve(i_num);
    f_attrs_.reserve(f_num);
    s_attrs_.reserve(s_num);
  }

  void Add(int64_t value) override {
    i_attrs_.emplace_back(value);
  }

  void Add(float value) override {
    f_attrs_.emplace_back(value);
  }

  void Add(std::string&& value) override {
    s_attrs_.emplace_back(std::move(value));
  }

  void Add(const std::string& value) override {
    s_attrs_.emplace_back(value);
  }

  void Add(const char* value, int32_t len) override {
    s_attrs_.emplace_back(value, len);
  }

  void Add(const int64_t* values, int32_t len) override {
    i_attrs_.assign(values, values + len);
  }

  void Add(const float* values, int32_t len) override {
    f_attrs_.assign(values, values + len);
  }

  const int64_t* GetInts(int32_t* len) const override {
    if (len) {
      *len = i_attrs_.size();
    }
    return i_attrs_.data();
  }

  const float* GetFloats(int32_t* len) const override {
    if (len) {
      *len = f_attrs_.size();
    }
    return f_attrs_.data();
  }

  const std::string* GetStrings(int32_t* len) const override {
    if (len) {
      *len = s_attrs_.size();
    }
    return s_attrs_.data();
  }

  const LiteString* GetLiteStrings(int32_t* len) const override {
    s_lites_.reserve(s_attrs_.size());
    for (auto& item : s_attrs_) {
      s_lites_.emplace_back(item.c_str(), item.length());
    }
    if (len) {
      *len = s_lites_.size();
    }
    return s_lites_.data();
  }

private:
  std::vector<int64_t>     i_attrs_;
  std::vector<float>       f_attrs_;
  std::vector<std::string> s_attrs_;
  mutable std::vector<LiteString> s_lites_;
};

class DataRefAttributeValue : public AttributeValue {
public:
  DataRefAttributeValue()
    : i_attrs_(nullptr), i_len_(0), f_attrs_(nullptr), f_len_(0) {
  }

  void Clear() override {
    i_attrs_ = nullptr;
    i_len_ = 0;
    f_attrs_ = nullptr;
    f_len_ = 0;
    s_lites_.clear();
    s_attrs_.clear();
  }

  void Shrink() override {
    s_lites_.shrink_to_fit();
  }

  void Swap(AttributeValue* rhs) override {
    DataRefAttributeValue* right = static_cast<DataRefAttributeValue*>(rhs);
    std::swap(i_attrs_, right->i_attrs_);
    std::swap(i_len_, right->i_len_);
    std::swap(f_attrs_, right->f_attrs_);
    std::swap(f_len_, right->f_len_);
    s_lites_.swap(right->s_lites_);
    s_attrs_.swap(right->s_attrs_);
  }

  void Reserve(int32_t i_num, int32_t f_num, int32_t s_num) override {
    s_attrs_.reserve(s_num);
  }

  void Add(int64_t value) override {
    // Not hold the value, just unimplement
  }

  void Add(float value) override {
    // Not hold the value, just unimplement
  }

  void Add(std::string&& value) override {
    // Not hold the value, just unimplement
  }

  void Add(const std::string& value) override {
    // Not hold the value, just unimplement
  }

  void Add(const char* value, int32_t len) override {
    s_lites_.emplace_back(value, len);
  }

  void Add(const int64_t* values, int32_t len) override {
    i_attrs_ = values;
    i_len_ = len;
  }

  void Add(const float* values, int32_t len) override {
    f_attrs_ = values;
    f_len_ = len;
  }

  const int64_t* GetInts(int32_t* len) const override {
    if (len) {
      *len = i_len_;
    }
    return i_attrs_;
  }

  const float* GetFloats(int32_t* len) const override {
    if (len) {
      *len = f_len_;
    }
    return f_attrs_;
  }

  const std::string* GetStrings(int32_t* len) const override {
    s_attrs_.reserve(s_lites_.size());
    for (auto& item : s_lites_) {
      s_attrs_.emplace_back(item.data(), item.size());
    }
    if (len) {
      *len = s_attrs_.size();
    }
    return s_attrs_.data();
  }

  const LiteString* GetLiteStrings(int32_t* len) const override {
    if (len) {
      *len = s_lites_.size();
    }
    return s_lites_.data();
  }

private:
  const int64_t* i_attrs_;
  int32_t        i_len_;
  const float*   f_attrs_;
  int32_t        f_len_;
  std::vector<LiteString> s_lites_;
  mutable std::vector<std::string> s_attrs_;
};

AttributeValue* NewDataHeldAttributeValue() {
  return new DataHeldAttributeValue();
}

AttributeValue* NewDataRefAttributeValue() {
  return new DataRefAttributeValue();
}

}  // namespace io
}  // namespace graphlearn
