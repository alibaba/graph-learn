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

#ifndef GRAPHLEARN_CORE_OPERATOR_GRAPH_NODE_GENERATOR_H_
#define GRAPHLEARN_CORE_OPERATOR_GRAPH_NODE_GENERATOR_H_

#include <algorithm>
#include <memory>
#include <random>
#include <unordered_map>

#include "common/threading/sync/lock.h"
#include "core/operator/utils/storage_wrapper.h"
#include "include/config.h"

namespace graphlearn {
namespace op {

class State {
public:
  State() : cursor_(0), epoch_(0) {
  }

  void Inc() {
    ++cursor_;
  }

  void Inc(int32_t delta) {
    cursor_ += delta;
  }

  int32_t Now() {
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
  int32_t cursor_;
  int32_t epoch_;
};

typedef std::shared_ptr<State> StatePtr;

class StateMap {
public:
  StatePtr GetState(const std::string& type, NodeFrom node_from) {
    if (states_[type][node_from]) {
      return states_[type][node_from];
    }
    states_[type][node_from].reset(new State);
    return states_[type][node_from];
  }

private:
  std::unordered_map<std::string, std::unordered_map<int32_t, StatePtr>>
      states_;
};


class Generator {
public:
  explicit Generator(StorageWrapper* storage) : storage_(storage),
                                                ids_(storage->GetIds()) {
  }
  virtual ~Generator() {
    delete storage_;
  };
  virtual bool Next(::graphlearn::io::IdType* ret) = 0;
  virtual void Reset() {}
  virtual void IncEpoch() {}
  virtual int32_t Epoch() {
    return 0;
  }

protected:
  StorageWrapper*  storage_;
  const ::graphlearn::io::IdArray ids_;
};

class RandomGenerator : public Generator {
public:
  explicit RandomGenerator(StorageWrapper* storage)
      : Generator(storage), dist_(0, ids_.Size() - 1) {
  }
  virtual ~RandomGenerator() = default;

  bool Next(::graphlearn::io::IdType* ret) override {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());
    int32_t rand = dist_(engine);
    *ret = ids_[rand];
    return true;
  }

private:
  std::uniform_int_distribution<int32_t> dist_;
};

class OrderedGenerator : public Generator {
public:
  explicit OrderedGenerator(StorageWrapper* storage) : Generator(storage) {
    state_ = GetState(storage_->Type(), storage_->From());
    storage_->Lock();
  }
  virtual ~OrderedGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* ret) override {
    if (state_->Now() >= ids_.Size()) {
      return false;
    }

    *ret = ids_[state_->Now()];
    state_->Inc();
    return true;
  }

  void Reset() override {
    state_->Reset();
    state_->IncEpoch();
  }

  int32_t Epoch() override {
    return state_->Epoch();
  }

private:
  StatePtr GetState(const std::string& type, NodeFrom node_from) {
    static std::mutex mtx;
    static StateMap* states = new StateMap();
    ScopedLocker<std::mutex> _(&mtx);
    return states->GetState(type, node_from);
  }

  StatePtr state_;
};

class ShuffleBuffer {
public:
  ShuffleBuffer() : cursor_(0), size_(0) {}

  bool HasNext() const { return (cursor_ < size_); }

  int32_t Size() const { return size_; }

  ::graphlearn::io::IdType Next() { return buffer_[cursor_++]; }

  void Fill(::graphlearn::io::IdType start,
            ::graphlearn::io::IdType end,
            const ::graphlearn::io::IdArray ids) {
    buffer_.clear();
    cursor_ = 0;
    size_ = std::min(
      end - start, static_cast<int64_t>(GLOBAL_FLAG(ShuffleBufferSize)));
    if (size_ <= 0) {
      return;
    }

    buffer_.reserve(size_);
    for (int32_t i = 0; i < size_; ++i) {
      buffer_.emplace_back(ids[start + i]);
    }

    thread_local static std::random_device rd;
    thread_local static std::default_random_engine rng(rd());
    std::shuffle(buffer_.begin(), buffer_.end(), rng);
  }

private:
  int32_t cursor_;
  int32_t size_;
  std::vector<::graphlearn::io::IdType> buffer_;
};

typedef std::shared_ptr<ShuffleBuffer> ShuffleBufferPtr;

class ShuffledGenerator : public Generator {
public:
  explicit ShuffledGenerator(StorageWrapper* storage) : Generator(storage) {
    state_ = GetState(storage_->Type(), storage_->From());
    shuffle_buffer_ = GetBuffer(storage_->Type(), storage_->From());
    storage_->Lock();
  }
  virtual ~ShuffledGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* ret) override {
    if (!shuffle_buffer_->HasNext()) {
      shuffle_buffer_->Fill(state_->Now(), ids_.Size(), ids_);
      state_->Inc(shuffle_buffer_->Size());
    }
    if (shuffle_buffer_->Size() == 0) {
      return false;
    }
    *ret = shuffle_buffer_->Next();
    return true;
  }

  void Reset() override {
    state_->Reset();
    state_->IncEpoch();
  }

  int32_t Epoch() override {
    return state_->Epoch();
  }

private:
  StatePtr GetState(const std::string& type, NodeFrom node_from) {
    static std::mutex mtx;
    static StateMap* states = new StateMap();
    ScopedLocker<std::mutex> _(&mtx);
    return states->GetState(type, node_from);
  }

  ShuffleBufferPtr GetBuffer(const std::string& type,
                             NodeFrom node_from) {
    static std::mutex mtx;
    static std::unordered_map<std::string,
        std::unordered_map<int32_t, ShuffleBufferPtr>> shuffle_bufs;
    ScopedLocker<std::mutex> _(&mtx);
    if (shuffle_bufs[type][node_from]) {
      return shuffle_bufs[type][node_from];
    }
    shuffle_bufs[type][node_from].reset(new ShuffleBuffer);
    return shuffle_bufs[type][node_from];
  }

  StatePtr state_;
  ShuffleBufferPtr shuffle_buffer_;
};

}  // namespace op
}  // namespace graphlearn

#endif //  GRAPHLEARN_CORE_OPERATOR_GRAPH_NODE_GENERATOR_H_
