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

#include "core/dag/tape.h"

#include <cassert>
#include <time.h>
#include <utility>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "common/threading/sync/lock.h"
#include "core/dag/dag.h"
#include "include/config.h"

namespace graphlearn {

Tape::Tape(const Dag* dag)
    : id_(-1),
      size_(dag->Size()),
      epoch_(-1),
      faked_(false),
      ready_(false),
      refs_(dag->Size()),
      recordings_(dag->Size()) {
  sem_init(&cond_, 0, 0);
  for (auto& node : dag->Nodes()) {
    refs_[node->Id() - 1] = node->InDegree();
  }
}

Tape::~Tape() {
  sem_destroy(&cond_);
}

void Tape::Record(int32_t key, std::unique_ptr<OpResponse>& response) {
  recordings_[key - 1] = {
    std::move(response->tensors_), std::move(response->sparse_tensors_)};
}

void Tape::Record(int32_t key, TensorMap&& tensors) {
  recordings_[key - 1] = std::move(tensors);
}

TensorMap& Tape::Retrieval(int32_t key) {
  assert(key > 0 && key <= size_);
  return recordings_[key - 1];
}

bool Tape::IsReadyFor(const DagNode* node) {
  if (--refs_[node->Id() - 1] == 0) {
    return true;
  }
  return false;
}

void Tape::SetReady() {
  ready_ = true;
  sem_post(&cond_);
}

void Tape::Fake() {
  recordings_.clear();
  faked_ = true;
  sem_post(&cond_);
}

void Tape::WaitUntilFinished() {
  sem_wait(&cond_);
}

void Tape::SetId(int32_t id) {
  id_ = id;
}

void Tape::SetEpoch(int32_t epoch) {
  epoch_ = epoch;
}

TapeStore::TapeStore(int32_t capacity, const Dag* dag)
    : cap_(capacity), dag_(dag), epoch_(0),
      tape_indexes_(GLOBAL_FLAG(ClientCount)) {
  sem_init(&empty_, 0, capacity);
  sem_init(&occupied_, 0, 0);
  for (int32_t cid = 0; cid < GLOBAL_FLAG(ClientCount); ++cid) {
    tape_indexes_[cid] = -1;
  }
}

TapeStore::~TapeStore() {
  sem_destroy(&empty_);
  sem_destroy(&occupied_);
}

Tape* TapeStore::New() {
  return new Tape(dag_);
}

void TapeStore::WaitAndPush(
    Tape* tape,
    const std::function<bool()>& stop_callback) {
  tape->SetEpoch(epoch_);
  if (tape->IsFaked()) {
    ++epoch_;
  }
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  // Set timeout to 100ms.
  ts.tv_nsec += 100 * 1000 * 1000;
  while (sem_timedwait(&empty_, &ts) == -1) {
    if (stop_callback()) {
      break;
    }
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 100 * 1000 * 1000;
  }
  Push(tape);
  sem_post(&occupied_);
}

Tape* TapeStore::WaitAndPop(int32_t client_id) {
  sem_wait(&occupied_);
  Tape* ret = Pop(client_id);
  ret->WaitUntilFinished();
  sem_post(&empty_);
  return ret;
}

void TapeStore::Push(Tape* tape) {
  ScopedLocker<std::mutex> _(&mtx_);
  queue_.push(tape);
}

Tape* TapeStore::Pop(int32_t client_id) {
  ScopedLocker<std::mutex> _(&mtx_);
  Tape* ret = queue_.front();
  queue_.pop();

  // To ensure the index order is same as pop order
  ret->SetId(++tape_indexes_[client_id]);
  return ret;
}

TapeStorePtr GetTapeStore(int32_t dag_id) {
  static std::mutex mtx;
  static std::unordered_map<int32_t, TapeStorePtr> buf;

  ScopedLocker<std::mutex> _(&mtx);
  if (buf[dag_id]) {
    return buf[dag_id];
  }
  Dag* dag = DagFactory::GetInstance()->Lookup(dag_id);
  if (dag == nullptr) {
    LOG(ERROR) << "GetTapeStore with not existed dag " << dag_id;
    return nullptr;
  }
  buf[dag_id].reset(new TapeStore(GLOBAL_FLAG(TapeCapacity), dag));
  return buf[dag_id];
}

}  // namespace graphlearn
