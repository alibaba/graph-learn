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

#ifndef GRAPHLEARN_INCLUDE_SHARDABLE_H_
#define GRAPHLEARN_INCLUDE_SHARDABLE_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace graphlearn {

class Sticker {
public:
  explicit Sticker(int32_t capacity)
      : capacity_(capacity),
        size_(0) {
    values_.resize(capacity);
  }

  void Add(int32_t shard_id, int32_t sticker) {
    assert(shard_id >= 0 && shard_id < capacity_);
    ++size_;
    values_[shard_id].push_back(sticker);
  }

  const std::vector<int32_t>& At(int32_t shard_id) {
    assert(shard_id >= 0 && shard_id < capacity_);
    return values_[shard_id];
  }

  int32_t Size() const {
    return size_;
  }

  void CopyFrom(const Sticker& other) {
    capacity_ = other.capacity_;
    size_ = other.size_;
    values_ = other.values_;
  }

public:
  int32_t capacity_;
  int32_t size_;
  std::vector<std::vector<int32_t>> values_;
};

/// A class used to keep the pieces that come from a shardable entity
/// after partitioned. Besides, it also contains the information to
/// recover the object before partitioned. These information is called
/// `Sticker`. Each piece in a `Shards` object is indexed by a `id`, and
/// the piece may contain more than one `Stitcher`.
template <class T>
class Shards {
public:
  explicit Shards(int32_t capacity)
      : capacity_(capacity),
        size_(0),
        cursor_(0),
        sticker_(new Sticker(capacity)) {
    ownership_.resize(capacity, false);
    pieces_.resize(capacity, nullptr);
  }

  ~Shards() {
    delete sticker_;

    for (int32_t i = 0; i < capacity_; ++i) {
      if (ownership_[i]) {
        delete pieces_[i];
      }
    }
  }

  /// Add a piece of T with corresponding id.
  /// If own=true, it will take the ownership of the added T.
  void Add(int32_t shard_id, T* t, bool own) {
    assert(shard_id >= 0 && shard_id < capacity_);
    pieces_[shard_id] = t;
    ownership_[shard_id] = own;
    ++size_;
  }

  /// Add a sticker to a given piece.
  T* AddSticker(int32_t shard_id, int32_t sticker) {
    sticker_->Add(shard_id, sticker);
    return pieces_[shard_id];
  }

  int32_t Capacity() const {
    return capacity_;
  }

  /// Return the count of pieces.
  int32_t Size() const {
    return size_;
  }

  Sticker* StickerPtr() {
    return sticker_;
  }

  /// Get reference of the next piece and move the cursor.
  /// If no more exist, return false. Otherwise, return true.
  bool Next(int32_t* shard_id, T** t) {
    while (cursor_ < capacity_) {
      if (pieces_[cursor_] == nullptr) {
        ++cursor_;
      } else {
        break;
      }
    }

    if (cursor_ >= capacity_) {
      return false;
    }

    *shard_id = cursor_;
    *t = pieces_[cursor_];
    ++cursor_;
    return true;
  }

  void ResetNext() {
    cursor_ = 0;
  }

  /// Move the ownership of the pieces.
  void MoveTo(std::vector<T*>* dst) {
    *dst = pieces_;
    ownership_.assign(capacity_, false);
  }

private:
  int32_t capacity_;
  int32_t size_;
  int32_t cursor_;
  std::vector<bool> ownership_;
  std::vector<T*>   pieces_;
  Sticker*          sticker_;
};

template <class T>
using ShardsPtr = std::shared_ptr<Shards<T>>;

template <class T>
class Shardable {
public:
  virtual ~Shardable() {}
  virtual ShardsPtr<T> Partition() const = 0;
};

template <class T>
class Joinable {
public:
  virtual ~Joinable() {}
  virtual void Stitch(ShardsPtr<T> shards) = 0;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SHARDABLE_H_
