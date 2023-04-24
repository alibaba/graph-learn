/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_OPERATOR_GRAPH_EDGE_GENERATOR_H_
#define GRAPHLEARN_CORE_OPERATOR_GRAPH_EDGE_GENERATOR_H_

#include <algorithm>
#include <memory>
#include <random>
#include <unordered_map>

#include "common/base/errors.h"
#include "common/threading/sync/lock.h"
#include "core/graph/graph_store.h"
#include "core/graph/storage/graph_storage.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

class EdgeState {
public:
  EdgeState() : cursor_(0), epoch_(0) {
  }

  void Inc() {
    ++cursor_;
  }

  void Inc(int32_t delta) {
    cursor_ += delta;
  }

  ::graphlearn::io::IdType Now() {
    return cursor_;
  }

  void Reset() {
    cursor_ = 0;
  }

  int32_t Epoch() {
    return epoch_;
  }

  void IncEpoch() {
    ++epoch_;
  }

  void Save() {
    // For persistence
  }

  void Load() {
    // For persistence
  }

private:
  ::graphlearn::io::IdType cursor_;
  int32_t epoch_;
};

typedef std::shared_ptr<EdgeState> StatePtr;

class EdgeGenerator {
public:
  explicit EdgeGenerator(::graphlearn::io::GraphStorage* storage)
      : storage_(storage) {
    edge_count_ = storage_->GetEdgeCount();
  }
  virtual ~EdgeGenerator() = default;

  virtual bool Next(::graphlearn::io::IdType* src_id,
                    ::graphlearn::io::IdType* dst_id,
                    ::graphlearn::io::IdType* edge_id) = 0;

  virtual void Reset() {}
  virtual void IncEpoch() {}
  virtual int32_t Epoch() {
    return 0;
  }

protected:
  ::graphlearn::io::GraphStorage* storage_;
  ::graphlearn::io::IdType        edge_count_;
};

class RandomEdgeGenerator : public EdgeGenerator {
public:
  explicit RandomEdgeGenerator(::graphlearn::io::GraphStorage* storage)
      : EdgeGenerator(storage), dist_(0, edge_count_ - 1) {
  }
  virtual ~RandomEdgeGenerator() = default;

  bool Next(::graphlearn::io::IdType* src_id,
            ::graphlearn::io::IdType* dst_id,
            ::graphlearn::io::IdType* edge_id) override {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());
    *edge_id = dist_(engine);
    *src_id = storage_->GetSrcId(*edge_id);
    *dst_id = storage_->GetDstId(*edge_id);
    return true;
  }

private:
  std::uniform_int_distribution<::graphlearn::io::IdType> dist_;
};

class OrderedEdgeGenerator : public EdgeGenerator {
public:
  explicit OrderedEdgeGenerator(::graphlearn::io::GraphStorage* storage)
      : EdgeGenerator(storage) {
    state_ = GetState(storage_->GetSideInfo()->type);
    storage_->Lock();
  }
  virtual ~OrderedEdgeGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* src_id,
            ::graphlearn::io::IdType* dst_id,
            ::graphlearn::io::IdType* edge_id) override {
    if (state_->Now() >= edge_count_) {
      return false;
    }

    auto edge_index = state_->Now();
    *edge_id = storage_->GetEdgeId(edge_index);
    *src_id = storage_->GetSrcId(*edge_id);
    *dst_id = storage_->GetDstId(*edge_id);
    state_->Inc();
    return true;
  }

  void Reset() override {
    state_->Reset();
  }

  void IncEpoch() override {
    state_->IncEpoch();
  }

  int32_t Epoch() override {
    return state_->Epoch();
  }

private:
  StatePtr GetState(const std::string& type) {
    static std::unordered_map<std::string, StatePtr> states;
    ScopedLocker<std::mutex> _(&mtx_);
    if (states[type]) {
      return states[type];
    }
    states[type].reset(new EdgeState);
    return states[type];
  }

  StatePtr state_;
  std::mutex mtx_;
};

class EdgeShuffleBuffer {
public:
  EdgeShuffleBuffer()
      : cursor_(0), size_(0) { }

  bool HasNext() const { return (cursor_ < size_); }

  int32_t Size() const { return size_; }

  ::graphlearn::io::IdType Next() { return buffer_[cursor_++]; }

  void Fill(::graphlearn::io::IdType start,
            ::graphlearn::io::IdType end) {
    ScopedLocker<std::mutex> _(&mtx_);
    buffer_.clear();
    cursor_ = 0;
    size_ = std::min(
      end - start, static_cast<int64_t>(GLOBAL_FLAG(ShuffleBufferSize)));
    if (size_ <= 0) {
      return;
    }

    buffer_.reserve(size_);
    for (int32_t i = 0; i < size_; ++i) {
      buffer_.emplace_back(start + i);
    }

    thread_local static std::random_device rd;
    thread_local static std::default_random_engine rng(rd());
    std::shuffle(buffer_.begin(), buffer_.end(), rng);
  }

private:
  int32_t cursor_;
  int32_t size_;
  std::mutex mtx_;
  std::vector<::graphlearn::io::IdType> buffer_;
};

typedef std::shared_ptr<EdgeShuffleBuffer> ShuffleBufferPtr;

class ShuffledEdgeGenerator : public EdgeGenerator {
public:
  explicit ShuffledEdgeGenerator(::graphlearn::io::GraphStorage* storage)
      : EdgeGenerator(storage) {
    state_ = GetState(storage_->GetSideInfo()->type);
    shuffle_buffer_ = GetBuffer(storage_->GetSideInfo()->type);
    storage_->Lock();
  }
  virtual ~ShuffledEdgeGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* src_id,
            ::graphlearn::io::IdType* dst_id,
            ::graphlearn::io::IdType* edge_id) override {
    if (!shuffle_buffer_->HasNext()) {
      shuffle_buffer_->Fill(state_->Now(), edge_count_);
      state_->Inc(shuffle_buffer_->Size());
    }
    if (shuffle_buffer_->Size() == 0) {
      return false;
    }
    *edge_id = shuffle_buffer_->Next();
    *src_id = storage_->GetSrcId(*edge_id);
    *dst_id = storage_->GetDstId(*edge_id);
    return true;
  }

  void Reset() override {
    state_->Reset();
  }

  void IncEpoch() override {
    state_->IncEpoch();
  }

  int32_t Epoch() override {
    return state_->Epoch();
  }

private:
  StatePtr GetState(const std::string& type) {
    static std::unordered_map<std::string, StatePtr> states;
    ScopedLocker<std::mutex> _(&mtx_);
    if (states[type]) {
      return states[type];
    }
    states[type].reset(new EdgeState);
    return states[type];
  }

  ShuffleBufferPtr GetBuffer(const std::string& type) {
    static std::unordered_map<std::string, ShuffleBufferPtr> shuffle_bufs;
    ScopedLocker<std::mutex> _(&mtx_);
    if (shuffle_bufs[type]) {
      return shuffle_bufs[type];
    }
    shuffle_bufs[type].reset(new EdgeShuffleBuffer);;
    return shuffle_bufs[type];
  }

  StatePtr         state_;
  ShuffleBufferPtr shuffle_buffer_;
  std::mutex       mtx_;
};

}  // namespace op
}  // namespace graphlearn

#endif //  GRAPHLEARN_CORE_OPERATOR_GRAPH_EDGE_GENERATOR_H_