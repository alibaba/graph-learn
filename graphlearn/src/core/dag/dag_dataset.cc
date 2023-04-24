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

#include "include/dag_dataset.h"

#include <cassert>
#include <limits>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "common/threading/sync/semaphore_shim.h"
#include "include/config.h"

namespace graphlearn {

Dataset::Dataset(Client* client, int32_t dag_id)
    : client_(client),
      dag_id_(dag_id),
      cap_(GLOBAL_FLAG(DatasetCapacity)),
      cursor_(0),
      head_(0),
      buffer_(cap_),
      occupied_(cap_) {
  assert(cap_ > 0 and cap_ < 128);
  tp_.reset(new ThreadPool(cap_));
  tp_->Startup();
  for (int32_t idx = 0; idx < cap_; ++idx) {
    sem_init(&(occupied_[idx]), 0, 0);
  }
  for (int32_t idx = 0; idx < cap_; ++idx) {
    PrefetchAsync();
  }
}

Dataset::~Dataset() {
  for (int32_t i = 0; i < cap_; ++i) {
    sem_destroy(&(occupied_[i]));
  }
}

void Dataset::Close() {
  if (tp_) {
    tp_->Shutdown();
  }
}

GetDagValuesResponse* Dataset::Next(int32_t epoch) {
  // Allowed lateness: 100s.
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += GLOBAL_FLAG(Timeout);
  if (sem_timedwait(&(occupied_[cursor_]), &ts) == -1) {
    LOG(ERROR) << "Query timeout. Try to increase timeout with `gl.set_timeout()`.";
    USER_LOG("Query timeout.");
    PrefetchAsync();
    ++cursor_ %= cap_;
    return Next(epoch);
  }

  auto ret = buffer_[cursor_];

  /// When multiple clients call `GetDagValues` from a single server,
  /// the responses of requested epoch may have been consumed
  /// by other clients.
  if (epoch < ret->Epoch()) {
    LOG(ERROR) << "Epoch " << epoch << " out of range.";
    USER_LOG("Out of range:No more data exist.");
    sem_post(&(occupied_[cursor_]));
    return nullptr;
  }

  buffer_[cursor_] = nullptr;
  PrefetchAsync();
  ++cursor_ %= cap_;
  return ret;
}

void Dataset::PrefetchAsync() {
  ++head_;
  tp_->AddTask(NewClosure(this, &Dataset::PrefetchFn));
}

void Dataset::PrefetchFn() {
  std::unique_ptr<GetDagValuesRequest> req(new GetDagValuesRequest(dag_id_));
  GetDagValuesResponse* res = new GetDagValuesResponse();
  Status s = client_->GetDagValues(req.get(), res);
  if (!s.ok()) {
    USER_LOG("Client fetch Dataset failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Client fetch Dataset failed: " << s.ToString();
    ::exit(-1);
  }

  int32_t index = res->Index();
  int32_t idx = index % cap_;

  if (head_ - index > cap_) {
    delete res;
    LOG(ERROR) << "Drop the obsoleted response with index " << index;
  } else if (buffer_[idx]) {
    /// To ensure that buffer[idx] is empty for accept a Response.
    delete res;
    LOG(ERROR) << "Dataset buffer[" << idx << "] is occupied";
  } else {
    buffer_[idx] = res;
    sem_post(&(occupied_[idx]));
  }
}

}  // namespace graphlearn
