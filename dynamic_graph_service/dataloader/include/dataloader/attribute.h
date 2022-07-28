/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DATALOADER_ATTRIBUTE_H_
#define DATALOADER_ATTRIBUTE_H_

#include <memory>
#include <string>
#include <vector>

#include "dataloader/typedefs.h"

namespace dgs {
namespace dataloader {

/// Self-managed byte buffer for storing value bytes.
class AttrByteBuf {
public:
  /// Creates an empty \c AttrByteBuf that does not point at anything.
  AttrByteBuf() = default;

  /// Creates an \c AttrByteBuf containing a copy of the provided data
  ///
  /// \param src  data buffer to be copied
  /// \param size size of data buffer in `src`
  AttrByteBuf(const int8_t* src, size_t size) {
    std::vector<int8_t> buf{src, src + size};
    auto d = std::make_unique<ObjectDeleter<std::vector<int8_t>>>(std::move(buf));
    data_ = d->obj.data();
    size_ = d->obj.size();
    deleter_ = std::move(d);
  }

  /// Creates an \c AttrByteBuf from a moved string.
  AttrByteBuf(std::string&& str) {
    auto d = std::make_unique<ObjectDeleter<std::string>>(std::move(str));
    data_ = reinterpret_cast<const int8_t*>(d->obj.data());
    size_ = d->obj.size();
    deleter_ = std::move(d);
  }

  /// Creates an \c AttrByteBuf containing a copy of c-styled string
  AttrByteBuf(const char* cstr): AttrByteBuf(std::string{cstr}) {}

  /// Creates an \c AttrByteBuf from an arithmetic value
  template <typename T, typename = typename std::enable_if_t<std::is_arithmetic<T>::value, bool>>
  inline
  AttrByteBuf(T val) {
    auto d = std::make_unique<ObjectDeleter<T>>(std::move(val));
    data_ = reinterpret_cast<const int8_t*>(&(d->obj));
    size_ = sizeof(T);
    deleter_ = std::move(d);
  }

  /// Creates an \c AttrByteBuf from a moved arithmetic value vector
  template <typename T, typename = typename std::enable_if_t<std::is_arithmetic<T>::value, bool>>
  AttrByteBuf(std::vector<T>&& vec) {
    auto d = std::make_unique<ObjectDeleter<std::vector<T>>>(std::move(vec));
    data_ = reinterpret_cast<const int8_t*>(d->obj.data());
    size_ = d->obj.size() * sizeof(T);
    deleter_ = std::move(d);
  }

  /// Disable copy constructor.
  AttrByteBuf(const AttrByteBuf&) = delete;

  /// Moves an \c AttrByteBuf.
  AttrByteBuf(AttrByteBuf&& x) noexcept : data_(x.data_), size_(x.size_), deleter_(std::move(x.deleter_)) {
    x.data_ = nullptr;
    x.size_ = 0;
  }

  ~AttrByteBuf() = default;

  /// Disable copy assignment.
  AttrByteBuf& operator=(AttrByteBuf&) = delete;

  /// Moves an \c AttrByteBuf.
  AttrByteBuf& operator=(AttrByteBuf&& x) noexcept {
    if (this != &x) {
      data_ = x.data_;
      size_ = x.size_;
      deleter_ = std::move(x.deleter_);
      x.data_ = nullptr;
      x.size_ = 0;
    }
    return *this;
  }

  /// Gets a pointer to the beginning of the byte buffer.
  const int8_t* Data() const {
    return data_;
  }

  /// Gets the byte buffer size.
  size_t Size() const {
    return size_;
  }

private:
  const int8_t* data_ = nullptr;
  size_t size_ = 0;

  struct Deleter {
    virtual ~Deleter() = default;
  };

  template <typename T, typename = typename std::enable_if_t<std::is_move_assignable<T>::value, bool>>
  struct ObjectDeleter final : Deleter {
    T obj;
    explicit ObjectDeleter(T&& obj) : obj(std::move(obj)) {}
    ObjectDeleter(const ObjectDeleter&) = delete;
    ObjectDeleter& operator=(ObjectDeleter&) = delete;
    /// Moves an object deleter.
    ObjectDeleter(ObjectDeleter&& x) noexcept = default;
    ObjectDeleter& operator=(ObjectDeleter&& x) noexcept = default;
  };

  std::unique_ptr<Deleter> deleter_;
};

struct AttrInfo {
  AttributeType attr_type = 0;
  AttributeValueType value_type = INT32;
  AttrByteBuf value_bytes;

  AttrInfo() = default;
  AttrInfo(AttributeType attr_type, AttributeValueType value_type, AttrByteBuf&& value_bytes)
    : attr_type(attr_type), value_type(value_type), value_bytes(std::move(value_bytes)) {}
};

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_ATTRIBUTE_H_
