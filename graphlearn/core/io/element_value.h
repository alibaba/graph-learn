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

#ifndef GRAPHLEARN_CORE_IO_ELEMENT_VALUE_H_
#define GRAPHLEARN_CORE_IO_ELEMENT_VALUE_H_

#include <string>
#include <utility>
#include <vector>
#include "graphlearn/common/io/value.h"

namespace graphlearn {
namespace io {

struct SideInfo {
  int32_t i_num;
  int32_t f_num;
  int32_t s_num;
  int32_t format;
  std::string type;
  std::string src_type;
  std::string dst_type;
  Direction direction;

  SideInfo()
      : i_num(0),
        f_num(0),
        s_num(0),
        format(0),
        direction(kOrigin) {
  }

  bool IsInitialized() const { return format != 0; }

  bool IsWeighted() const { return format & kWeighted; }

  bool IsLabeled() const { return format & kLabeled; }

  bool IsAttributed() const { return format & kAttributed; }

  void SetWeighted() { format &= kWeighted; }

  void SetLabeled() { format &= kLabeled; }

  void SetAttributed() { format &= kAttributed; }

  void CopyFrom(const SideInfo& info) {
    i_num = info.i_num;
    f_num = info.f_num;
    s_num = info.s_num;
    format = info.format;
    type = info.type;
    src_type = info.src_type;
    dst_type = info.dst_type;
    direction = info.direction;
  }
};

struct AttributeValue {
  std::vector<int64_t>     i_attrs;
  std::vector<float>       f_attrs;
  std::vector<std::string> s_attrs;

  AttributeValue() {}

  AttributeValue(const AttributeValue& right) {
    i_attrs = right.i_attrs;
    f_attrs = right.f_attrs;
    s_attrs = right.s_attrs;
  }

  static AttributeValue* Default(const SideInfo* info);

  void Reserve(int32_t i_num, int32_t f_num, int32_t s_num) {
    i_attrs.reserve(i_num);
    f_attrs.reserve(f_num);
    s_attrs.reserve(s_num);
  }

  void AddInt(int64_t i) {
    i_attrs.emplace_back(i);
  }

  void AddFloat(float f) {
    f_attrs.emplace_back(f);
  }

  void AddString(std::string& s) {  // NOLINT
    s_attrs.emplace_back(std::move(s));
  }

  void Clear() {
    i_attrs.clear();
    f_attrs.clear();
    s_attrs.clear();
  }
};

struct EdgeValue : public AttributeValue {
  int64_t src_id;
  int64_t dst_id;
  float   weight;
  int32_t label;
};

struct NodeValue : public AttributeValue {
  int64_t id;
  float   weight;
  int32_t label;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_ELEMENT_VALUE_H_
