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

#ifndef GRAPHLEARN_CORE_DAG_TAPE_H_
#define GRAPHLEARN_CORE_DAG_TAPE_H_

#include <semaphore.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "graphlearn/include/op_request.h"

namespace graphlearn {

class Dag;
class DagNode;

class Tape {
public:
  Tape(const Dag* dag);
  ~Tape();

  int32_t Size() { return size_; }

  /// Write a record on the tape.
  void Record(int32_t key, std::unique_ptr<OpResponse>& response);
  void Record(int32_t key, Tensor::Map&& tensors);

  /// Lookup record with given key. If not found, return nullptr.
  const Tensor::Map& Retrieval(int32_t key);

  /// All the input data for running a DagNode will be dumped to a tape.
  /// Check whether it is ready for the given node.
  bool IsReadyFor(const DagNode* node);

  /// Mark the tape ready. That means all the outputs of the nodes have
  /// been collected correctly.
  void SetReady();

  /// Mark the tape with failed status.
  void Fake();

  /// Wait until the tape is ready or faked.
  void WaitUntilFinished();

  /// The order of tape.
  void SetId(int32_t id);

  /// The dag is executed asynchronously and parallelly. We need to care
  /// the EPOCH info in case of misorder of the generated data.
  void SetEpoch(int32_t epoch);

  bool IsReady() const {
    return ready_;
  }

  bool IsFaked() const {
    return faked_;
  }

  int32_t Id() const {
    return id_;
  }

  int32_t Epoch() const {
    return epoch_;
  }

private:
  int32_t id_;
  int32_t size_;

  std::atomic<bool> faked_;
  std::atomic<bool> ready_;

  sem_t      cond_;
  std::atomic<int32_t> epoch_;
  // DagNode with Id i records on index i-1
  std::vector<Tensor::Map> recordings_;
  std::vector<std::atomic<int32_t>> refs_;
};

class TapeStore {
public:
  explicit TapeStore(int32_t capacity, const Dag* dag);
  ~TapeStore();

  /// Create a new tape without data, which can be used for writing.
  /// This interface will be used by DagScheduler, to record dag node
  /// values on.
  Tape* New();

  /// Push the tape into FIFO-Queue until succeed or break with stop_callback
  /// when timeout.
  void WaitAndPush(Tape* tape, const std::function<bool()>& stop_callback);

  /// Pop a ready or faked tape until succeed.
  Tape* WaitAndPop(int32_t client_id);

private:
  void Push(Tape* tape);
  Tape* Pop(int32_t client_id);

private:
  sem_t      empty_;
  sem_t      occupied_;
  int32_t    cap_;
  int32_t    epoch_;
  const Dag* dag_;
  std::mutex mtx_;

  std::queue<Tape*> queue_;
  // Record the tape order for each client.
  std::vector<std::atomic<int32_t>> tape_indexes_;
};

typedef std::shared_ptr<TapeStore> TapeStorePtr;
TapeStorePtr GetTapeStore(int32_t dag_id);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_DAG_TAPE_H_
