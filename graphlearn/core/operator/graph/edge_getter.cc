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

#include <algorithm>
#include <memory>
#include <random>
#include <unordered_map>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/graph_request.h"

namespace graphlearn {
namespace op {

namespace {

class State {
public:
  State() : cursor_(0) {
  }

  void Inc() {
    ++cursor_;
  }

  void Inc(int32_t step_size) {
    cursor_ += step_size;
  }

  ::graphlearn::io::IdType Now() {
    return cursor_;
  }

  void Reset() {
    cursor_ = 0;
  }

  void Save() {
    // For persistence
  }

  void Load() {
    // For persistence
  }

private:
  ::graphlearn::io::IdType cursor_;
};

typedef std::shared_ptr<State> StatePtr;

class Generator {
public:
  explicit Generator(::graphlearn::io::GraphStorage* storage)
      : storage_(storage) {
    edge_count_ = storage_->GetEdgeCount();
  }
  virtual ~Generator() = default;

  virtual bool Next(::graphlearn::io::IdType* src_id,
                    ::graphlearn::io::IdType* dst_id,
                    ::graphlearn::io::IdType* edge_id) = 0;

  virtual void Reset() {}

protected:
  ::graphlearn::io::GraphStorage* storage_;
  ::graphlearn::io::IdType        edge_count_;
};

class RandomGenerator : public Generator {
public:
  explicit RandomGenerator(::graphlearn::io::GraphStorage* storage)
      : Generator(storage), dist_(0, edge_count_ - 1) {
  }
  virtual ~RandomGenerator() = default;

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

class OrderedGenerator : public Generator {
public:
  explicit OrderedGenerator(::graphlearn::io::GraphStorage* storage)
      : Generator(storage) {
    state_ = GetState(storage_->GetSideInfo()->type);
    storage_->Lock();
  }
  virtual ~OrderedGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* src_id,
            ::graphlearn::io::IdType* dst_id,
            ::graphlearn::io::IdType* edge_id) override {
    if (state_->Now() >= edge_count_) {
      return false;
    }

    *edge_id = state_->Now();
    *src_id = storage_->GetSrcId(*edge_id);
    *dst_id = storage_->GetDstId(*edge_id);
    state_->Inc();
    return true;
  }

  void Reset() override {
    state_->Reset();
  }

private:
  StatePtr GetState(const std::string& type) {
    static std::mutex mtx;
    static std::unordered_map<std::string, StatePtr> states;
    ScopedLocker<std::mutex> _(&mtx);
    if (states[type]) {
      return states[type];
    }
    states[type].reset(new State);
    return states[type];
  }

  StatePtr state_;
};

class ShuffleBuffer {
public:
  ShuffleBuffer()
      : cursor_(0), size_(0) { }

  bool HasNext() const { return (cursor_ < size_); }

  int32_t Size() const { return size_; }

  ::graphlearn::io::IdType Next() { return buffer_[cursor_++]; }

  void Fill(::graphlearn::io::IdType start,
            ::graphlearn::io::IdType end) {
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
  std::vector<::graphlearn::io::IdType> buffer_;
};

typedef std::shared_ptr<ShuffleBuffer> ShuffleBufferPtr;

class ShuffledGenerator : public Generator {
public:
  explicit ShuffledGenerator(::graphlearn::io::GraphStorage* storage)
      : Generator(storage) {
    state_ = GetState(storage_->GetSideInfo()->type);
    shuffle_buffer_ = GetBuffer(storage_->GetSideInfo()->type);
    storage_->Lock();
  }
  virtual ~ShuffledGenerator() {
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

private:
  StatePtr GetState(const std::string& type) {
    static std::mutex mtx;
    static std::unordered_map<std::string, StatePtr> states;
    ScopedLocker<std::mutex> _(&mtx);
    if (states[type]) {
      return states[type];
    }
    states[type].reset(new State);
    return states[type];
  }

  ShuffleBufferPtr GetBuffer(const std::string& type) {
    static std::mutex mtx;
    static std::unordered_map<std::string, ShuffleBufferPtr> shuffle_bufs;
    ScopedLocker<std::mutex> _(&mtx);
    if (shuffle_bufs[type]) {
      return shuffle_bufs[type];
    }
    shuffle_bufs[type].reset(new ShuffleBuffer);;
    return shuffle_bufs[type];
  }

  StatePtr state_;
  ShuffleBufferPtr shuffle_buffer_;
};

}  // anonymous namespace

class EdgeGetter : public RemoteOperator {
public:
  virtual ~EdgeGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetEdgesRequest* request =
      static_cast<const GetEdgesRequest*>(req);
    GetEdgesResponse* response =
      static_cast<GetEdgesResponse*>(res);

    Graph* graph = graph_store_->GetGraph(request->EdgeType());
    ::graphlearn::io::GraphStorage* storage = graph->GetLocalStorage();

    std::unique_ptr<Generator> generator;
    if (request->Strategy() == "by_order") {
      generator.reset(new OrderedGenerator(storage));
    } else if (request->Strategy() == "random") {
      generator.reset(new RandomGenerator(storage));
    } else {
      generator.reset(new ShuffledGenerator(storage));
    }

    ::graphlearn::io::IdType src_id, dst_id, edge_id;
    int32_t expect_size = request->BatchSize();
    response->Init(expect_size);
    for (int32_t i = 0; i < expect_size; ++i) {
      if (generator->Next(&src_id, &dst_id, &edge_id)) {
        response->Append(src_id, dst_id, edge_id);
      } else {
        break;
      }
    }

    if (response->Size() > 0) {
      return Status::OK();
    } else {
      // Begin next epoch.
      generator->Reset();
      return error::OutOfRange("No more edges exist.");
    }
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }
};

REGISTER_OPERATOR("GetEdges", EdgeGetter);

}  // namespace op
}  // namespace graphlearn
