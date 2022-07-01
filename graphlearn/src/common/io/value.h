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

#ifndef GRAPHLEARN_COMMON_IO_VALUE_H_
#define GRAPHLEARN_COMMON_IO_VALUE_H_

#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include "graphlearn/include/constants.h"

namespace graphlearn {
namespace io {

union NumericValue {
  int32_t i;
  int64_t l;
  float   f;
  double  d;
};

struct StringValue {
  const char*  data;
  size_t       len;
  bool         own;

  StringValue() : data(nullptr), len(0), own(false) {
  }

  StringValue(const char* d, size_t l, bool copy = false)
      : own(false) {
    Copy(d, l, copy);
  }

  explicit StringValue(const std::string& s, bool copy = false)
      : own(false) {
    Copy(s, copy);
  }

  ~StringValue() {
    if (own) {
      delete [] data;
    }
  }

  void Copy(const StringValue& r, bool copy) {
    Copy(r.data, r.len, copy);
  }

  void Copy(const std::string& r, bool copy) {
    Copy(r.c_str(), r.length(), copy);
  }

  void Copy(const char* d, size_t l, bool copy) {
    if (own) {
      delete [] data;
      data = nullptr;
    }

    own = copy;
    len = l;
    if (d == nullptr || !copy) {
      data = d;
    } else {
      data = new char[l + 1];
      char* non_const = const_cast<char*>(data);
      memcpy(non_const, d, l);
      non_const[l] = '\0';
    }
  }

  std::string ToString() const {
    return std::string(data, len);
  }
};

struct Value {
  NumericValue n;
  StringValue  s;

  Value() {}

  Value(const Value& r, bool copy = false) {
    n = r.n;
    s.Copy(r.s, copy);
  }

  void Copy(const Value& r, bool copy = false) {
    n = r.n;
    s.Copy(r.s, copy);
  }
};

struct Schema {
  std::vector<std::string> names;
  std::vector<DataType>    types;

  Schema() {}

  explicit Schema(const std::vector<DataType>& t) : types(t) {
    names.resize(types.size());
  }

  static DataType Translate(DataType t) {
    static DataType TransBitmap[] = {kInt32, kInt32, kFloat, kFloat, kString};
    return t < kUnknown ? TransBitmap[t] : kUnknown;
  }

  size_t Size() const {
    return names.size();
  }

  void Append(const std::string& name, DataType type) {
    names.push_back(name);
    types.push_back(type);
  }

  std::string ToString() const {
    std::stringstream ss;
    size_t size = Size();
    for (size_t i = 0; i < size; ++i) {
      DataType t = Translate(types[i]);
      if (t == kInt32) {
        ss << "int,";
      } else if (t == kFloat) {
        ss << "float,";
      } else if (t == kString) {
        ss << "string,";
      } else {
        ss << "unknown,";
      }
    }
    std::string ret = ss.str();
    if (!ret.empty()) {
      ret.pop_back();
    }
    return ret;
  }

  bool operator == (const Schema& right) {
    size_t size = Size();
    if (size != right.Size()) {
      return false;
    }

    for (size_t i = 0; i < size; ++i) {
      if (Translate(types[i]) != Translate(right.types[i])) {
        return false;
      }
    }
    return true;
  }

  bool operator != (const Schema& right) {
    return !(*this == right);
  }
};

struct Record {
  std::vector<Value> values;

  Record() {}

  Record(Record&& record) {
    Reserve(record.values.size());
    Swap(&record);
  }

  Record& operator = (Record&& record) {
    Reserve(record.values.size());
    Swap(&record);
    return *this;
  }

  void Clear() {
    values.clear();
  }

  void Swap(Record* record) {
    values.swap(record->values);
  }

  void Reserve(size_t num) {
    values.resize(num);
  }

  void Append(const Value& v) {
    values.push_back(v);
  }

  Value& operator[] (size_t i) {
    return values[i];
  }

  const Value& operator[] (size_t i) const {
    return values[i];
  }
};

DataType ToDataType(const std::string& type);

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_IO_VALUE_H_
