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
#include "graphlearn/core/graph/storage/node_storage.h"
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

  int32_t Now() {
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
  int32_t cursor_;
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

class DataStorage {
public:
  DataStorage(NodeFrom node_from, const std::string& type,
      GraphStore* graph_store) : node_from_(node_from) {
    if (node_from == NodeFrom::kNode) {
      Noder* noder = graph_store->GetNoder(type);
      node_storage_ = noder->GetLocalStorage();
      graph_storage_ = nullptr;
    } else {
      Graph* graph = graph_store->GetGraph(type);
      graph_storage_ = graph->GetLocalStorage();
      node_storage_ = nullptr;
    }
  }

  const ::graphlearn::io::IdList* GetIds() const {
    if (node_from_ == NodeFrom::kNode) {
      return node_storage_->GetIds();
    } else {
      if (node_from_ == NodeFrom::kEdgeSrc) {
        return graph_storage_->GetAllSrcIds();
      } else {
        return graph_storage_->GetAllDstIds();
      }
    }
  }

  void Lock() {
    if (node_storage_ != nullptr) {
      node_storage_->Lock();
    } else {
      graph_storage_->Lock();
    }
  }

  void Unlock() {
    if (node_storage_ != nullptr) {
      node_storage_->Unlock();
    } else {
      graph_storage_->Unlock();
    }
  }

  const std::string& Type() const {
    if (node_storage_ != nullptr) {
      return node_storage_->GetSideInfo()->type;
    } else {
      return graph_storage_->GetSideInfo()->type;
    }
  }

  NodeFrom From() const {
    return node_from_;
  }

private:
  ::graphlearn::io::NodeStorage*  node_storage_;
  ::graphlearn::io::GraphStorage* graph_storage_;
  NodeFrom                        node_from_;
};

class Generator {
public:
  explicit Generator(DataStorage* storage) : storage_(storage) {
    ids_ = storage_->GetIds();
  }
  virtual ~Generator() = default;
  virtual bool Next(::graphlearn::io::IdType* ret) = 0;
  virtual void Reset() {}

protected:
  DataStorage*  storage_;
  const ::graphlearn::io::IdList* ids_;
};

class RandomGenerator : public Generator {
public:
  explicit RandomGenerator(DataStorage* storage)
      : Generator(storage), dist_(0, ids_->size() - 1) {
  }
  virtual ~RandomGenerator() = default;

  bool Next(::graphlearn::io::IdType* ret) override {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());
    int32_t rand = dist_(engine);
    *ret = (*ids_)[rand];
    return true;
  }

private:
  std::uniform_int_distribution<int32_t> dist_;
};

class OrderedGenerator : public Generator {
public:
  explicit OrderedGenerator(DataStorage* storage) : Generator(storage) {
    state_ = GetState(storage_->Type(), storage_->From());
    storage_->Lock();
  }
  virtual ~OrderedGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* ret) override {
    if (state_->Now() >= ids_->size()) {
      return false;
    }

    *ret = (*ids_)[state_->Now()];
    state_->Inc();
    return true;
  }

  void Reset() override {
    state_->Reset();
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
            const ::graphlearn::io::IdList* ids) {
    buffer_.clear();
    cursor_ = 0;
    size_ = std::min(
      end - start, static_cast<int64_t>(GLOBAL_FLAG(ShuffleBufferSize)));
    if (size_ <= 0) {
      return;
    }

    buffer_.reserve(size_);
    for (int32_t i = 0; i < size_; ++i) {
      buffer_.emplace_back((*ids)[start + i]);
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
  explicit ShuffledGenerator(DataStorage* storage) : Generator(storage) {
    state_ = GetState(storage_->Type(), storage_->From());
    shuffle_buffer_ = GetBuffer(storage_->Type(), storage_->From());
    storage_->Lock();
  }
  virtual ~ShuffledGenerator() {
    storage_->Unlock();
  }

  bool Next(::graphlearn::io::IdType* ret) override {
    if (!shuffle_buffer_->HasNext()) {
      shuffle_buffer_->Fill(state_->Now(), ids_->size(), ids_);
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

}  // anonymous namespace

class NodeGetter : public RemoteOperator {
public:
  virtual ~NodeGetter() = default;

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const GetNodesRequest* request =
      static_cast<const GetNodesRequest*>(req);
    GetNodesResponse* response =
      static_cast<GetNodesResponse*>(res);

    DataStorage* storage = new DataStorage(
      request->GetNodeFrom(), request->Type(), graph_store_);
    std::unique_ptr<Generator> generator = GetGenerator(
      storage, request->Strategy());
    return GetNode(generator, request, response);
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }


private:
  std::unique_ptr<Generator> GetGenerator(DataStorage* storage,
                                          const std::string& strategy) {
    std::unique_ptr<Generator> generator;
    if (strategy == "by_order") {
      generator.reset(new OrderedGenerator(storage));
    } else if (strategy == "random") {
      generator.reset(new RandomGenerator(storage));
    } else {
      generator.reset(new ShuffledGenerator(storage));
    }
    return generator;
  }

  Status GetNode(const std::unique_ptr<Generator>& generator,
                 const GetNodesRequest* request,
                 GetNodesResponse* response) {
    ::graphlearn::io::IdType id = 0;
    int32_t expect_size = request->BatchSize();
    response->Init(expect_size);
    for (int32_t i = 0; i < expect_size; ++i) {
      if (generator->Next(&id)) {
        response->Append(id);
      } else {
        break;
      }
    }

    if (response->Size() > 0) {
      return Status::OK();
    } else {
      // Begin next epoch.
      generator->Reset();
      return error::OutOfRange("No more nodes exist.");
    }
  }
};

REGISTER_OPERATOR("GetNodes", NodeGetter);

}  // namespace op
}  // namespace graphlearn
