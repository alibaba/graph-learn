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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "core/io/element_value.h"

namespace graphlearn {
namespace io {

typedef int64_t IdType;
typedef int32_t IndexType;

typedef std::vector<IdType> IdList;
typedef std::vector<IndexType> IndexList;
typedef std::vector<std::vector<IdType>> IdMatrix;
typedef std::unordered_map<IdType, IndexType> MAP;

template <class T>
class Array {
public:
  Array() : value_(nullptr), size_(0) {
  }

  Array(const T* value, int32_t size)
    : value_(value), size_(size) {
  }

  Array(const T* value, int32_t size, const std::shared_ptr<T> ownership)
    : value_(value), size_(size), ownership_(ownership) {
  }

  explicit Array(const std::vector<T>& values)
    : value_(values.data()), size_(values.size()) {
  }

  Array(const Array& rhs) {
    value_ = rhs.value_;
    size_ = rhs.size_;
  }

  Array& operator=(const Array& rhs) {
    value_ = rhs.value_;
    size_ = rhs.size_;
    return *this;
  }

  Array(Array&& rhs) {
    value_ = rhs.value_;
    size_ = rhs.size_;
  }

  virtual operator bool () const {
    return value_ != nullptr && size_ != 0;
  }

  virtual T operator[] (int32_t i) const {
    return value_[i];
  }

  T at(int32_t i) const {
    return this->operator[](i);
  }

  virtual int32_t Size() const {
    return size_;
  }

  virtual const T *data() const {
    return value_;
  }

private:
  const T* value_;
  int32_t size_;

  // the `ownership` pointer is to keep the underlying data alive until is array
  // itself is not used anymore, without copying data around.
  std::shared_ptr<T> ownership_;
};

typedef Array<IdType> IdArray;
typedef Array<IndexType> IndexArray;

class Attribute {
public:
  Attribute() : value_(nullptr), own_(false) {
  }

  Attribute(AttributeValue* value, bool own)
    : value_(value), own_(own) {
  }

  Attribute(Attribute&& rhs) {
    value_ = rhs.value_;
    own_ = rhs.own_;
    rhs.value_ = nullptr;
    rhs.own_ = false;
  }

  ~Attribute() {
    if (own_) {
      delete value_;
    }
  }

  AttributeValue* operator -> () const {
    return value_;
  }

  AttributeValue* get() const {
    return value_;
  }

private:
  AttributeValue* value_;
  bool own_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_TYPES_H_

