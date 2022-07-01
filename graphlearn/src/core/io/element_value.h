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

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graphlearn/common/io/value.h"
#include "graphlearn/common/string/lite_string.h"

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

class AttributeValue {
public:
  static AttributeValue* Default(const SideInfo* info);

  virtual void Clear() = 0;
  virtual void Shrink() = 0;
  virtual void Swap(AttributeValue* rhs) = 0;
  virtual void Reserve(int32_t i, int32_t f, int32_t s) = 0;

  virtual void Add(int64_t value) = 0;
  virtual void Add(float value) = 0;
  virtual void Add(std::string&& value) = 0;
  virtual void Add(const std::string& value) = 0;
  virtual void Add(const char* value, int32_t len) = 0;
  virtual void Add(const int64_t* values, int32_t len) = 0;
  virtual void Add(const float* values, int32_t len) = 0;
  virtual const int64_t* GetInts(int32_t* len) const = 0;
  virtual const float* GetFloats(int32_t* len) const = 0;
  virtual const std::string* GetStrings(int32_t* len) const = 0;
  virtual const LiteString* GetLiteStrings(int32_t* len) const = 0;
};

AttributeValue* NewDataHeldAttributeValue();
AttributeValue* NewDataRefAttributeValue();

struct EdgeValue {
  int64_t src_id;
  int64_t dst_id;
  float   weight;
  int32_t label;
  AttributeValue* attrs;

  EdgeValue() : attrs(NewDataHeldAttributeValue()) {}
  ~EdgeValue() { delete attrs; }
};

struct NodeValue {
  int64_t id;
  float   weight;
  int32_t label;
  AttributeValue* attrs;

  NodeValue() : attrs(NewDataHeldAttributeValue()) {}
  ~NodeValue() { delete attrs; }
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_ELEMENT_VALUE_H_
