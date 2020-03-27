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

#include "graphlearn/common/rpc/notification.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/time_stamp.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/common/threading/sync/waitable_event.h"
#include "graphlearn/include/config.h"

namespace graphlearn {

class RpcNotificationImpl {
public:
  RpcNotificationImpl();
  ~RpcNotificationImpl() = default;

  void Init(const std::string& req_type,
            int32_t size);

  int32_t AddRpcTask(int32_t remote_id);

  void SetCallback(RpcNotification::Callback cb);

  void Notify(int32_t remote_id);

  void NotifyFail(int32_t remote_id, const Status& status);

  void Wait(int64_t timeout_ms = -1);

private:
  bool initialized;

  std::atomic<int32_t> total_tasks_;
  std::atomic<int32_t> finished_tasks_;
  std::atomic<int32_t> failed_tasks_;

  int64_t begin_time_;
  int64_t end_time_;

  std::string req_type_;
  RpcNotification::Callback cb_;

  typedef RWLock LockType;
  LockType lock;
  std::unordered_map<int32_t, int32_t> remote_ids_index_;
  std::vector<bool> finished_bitmap_;
  std::vector<int64_t> rpc_times_;

  WaitableEvent event_;
};

RpcNotification::RpcNotification() {
  impl_ = new RpcNotificationImpl;
}

RpcNotification::~RpcNotification() {
  delete impl_;
}

void RpcNotification::Init(const std::string& req_type,
                           int32_t size) {
  impl_->Init(req_type, size);
}

int32_t RpcNotification::AddRpcTask(int32_t remote_id) {
  return impl_->AddRpcTask(remote_id);
}

void RpcNotification::SetCallback(RpcNotification::Callback cb) {
  impl_->SetCallback(cb);
}

void RpcNotification::Notify(int32_t remote_id) {
  impl_->Notify(remote_id);
}

void RpcNotification::NotifyFail(int32_t remote_id, const Status& status) {
  impl_->NotifyFail(remote_id, status);
}

void RpcNotification::Wait(int64_t timeout_ms) {
  impl_->Wait(timeout_ms);
}

RpcNotificationImpl::RpcNotificationImpl()
  : initialized(false),
    total_tasks_(0),
    finished_tasks_(0),
    failed_tasks_(0),
    begin_time_(-1),
    end_time_(-1),
    cb_(nullptr) {
}

void RpcNotificationImpl::Init(const std::string& req_type,
                               int32_t size) {
  LockType::WriterLocker _(lock);
  if (!initialized) {
    req_type_ = req_type;
    total_tasks_.store(size);
    remote_ids_index_.reserve(size);
    finished_bitmap_.resize(size, false);
    rpc_times_.resize(size, 0);
    initialized = true;
    begin_time_ = GetTimeStampInUs();
    LOG(INFO) << "RpcNotification:Start"
              << "\treq_type:" << req_type_
              << "\tsize:" << size;
  }
}

int32_t RpcNotificationImpl::AddRpcTask(int32_t remote_id) {
  LockType::WriterLocker _(lock);
  if (remote_ids_index_.find(remote_id) == remote_ids_index_.end()) {
    int32_t index = remote_ids_index_.size();
    remote_ids_index_[remote_id] = index;
  }
  return remote_ids_index_.size();
}

void RpcNotificationImpl::SetCallback(RpcNotification::Callback cb) {
  LockType::WriterLocker _(lock);
  if (cb_ == nullptr) {
    cb_ = cb;
  }
}

void RpcNotificationImpl::Notify(int32_t remote_id) {
  LOG(INFO) << "RpcNotification:Notify"
            << "\treq_type:" << req_type_
            << "\tremote_id:" << remote_id
            << "\ttotal:" << total_tasks_.load();
  std::unordered_map<int32_t, int32_t>::const_iterator it;
  {
    LockType::ReaderLocker _(lock);
    it = remote_ids_index_.find(remote_id);
    if (it == remote_ids_index_.end() || finished_bitmap_[it->second]) {
      LOG(WARNING) << "RpcNotification:invalid_id"
                   << "\tremote_id:" << remote_id;
      return;
    }
  }
  int32_t index = it->second;
  finished_bitmap_[index] = true;
  rpc_times_[index] = (GetTimeStampInUs() - begin_time_) / 1000;
  int32_t finished = finished_tasks_.fetch_add(1);
  if (finished + 1 >= total_tasks_.load()) {
    LOG(INFO) << "RpcNotification:Done"
              << "\treq_type:" << req_type_;
    if (cb_ != nullptr) {
      cb_(req_type_, Status::OK());
    }
    event_.Set();
  }
}

void RpcNotificationImpl::NotifyFail(int32_t remote_id,
                                     const Status& status) {
  std::unordered_map<int32_t, int32_t>::const_iterator it;
  {
    LockType::ReaderLocker _(lock);
    it = remote_ids_index_.find(remote_id);
    if (it == remote_ids_index_.end() || finished_bitmap_[it->second]) {
      LOG(WARNING) << "RpcNotification:invalid_id"
                   << "\tremote_id:" << remote_id;
      return;
    }
  }
  int32_t index = it->second;
  finished_bitmap_[index] = true;
  rpc_times_[index] = (GetTimeStampInUs() - begin_time_) / 1000;
  int32_t finished = finished_tasks_.fetch_add(1);
  failed_tasks_.fetch_add(1);
  LOG(ERROR) << "RpcNotification:Failed"
             << "\treq_type:" << req_type_
             << "\tstatus:" << status.ToString();
  if (finished + 1 >= total_tasks_.load()) {
    LOG(WARNING) << "RpcNotification:Done"
                 << "\treq_type:" << req_type_;
    if (cb_ != nullptr) {
      cb_(req_type_, status);
    }
    event_.Set();
  }
}

void RpcNotificationImpl::Wait(int64_t timeout_ms) {
  if (total_tasks_.load() > 0) {
    if (!event_.Wait(timeout_ms)) {
      LOG(ERROR) << "RpcNotification:timeout"
                 << "\treq_type:" << req_type_;
      if (cb_ != nullptr) {
        cb_(req_type_, error::DeadlineExceeded("rpc timeout."));
      }
    }
  }
}

}  // namespace graphlearn
