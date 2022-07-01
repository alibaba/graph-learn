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

#include <unistd.h>
#include <chrono>  // NOLINT [build/c++11]
#include <memory>
#include <thread>  // NOLINT [build/c++11]
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/macros.h"
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/coordinator.h"

namespace graphlearn {

FSCoordinator::FSCoordinator(int32_t server_id, int32_t server_count,
                             Env* env)
    : Coordinator(server_id, server_count, env) {
  if (strings::EndWith(GLOBAL_FLAG(Tracker), "/")) {
    tracker_ = GLOBAL_FLAG(Tracker);
  } else {
    tracker_ = GLOBAL_FLAG(Tracker) + "/";
  }

  Status s = env->GetFileSystem(GLOBAL_FLAG(Tracker), &fs_);
  if (!s.ok()) {
    LOG(FATAL) << "Invalid tracker path: " << tracker_;
    ::exit(-1);
  }
  auto tp = env->ReservedThreadPool();
  tp->AddTask(NewClosure(this, &FSCoordinator::Refresh));
}

void FSCoordinator::Finallize() {
  auto tp = Env::Default()->ReservedThreadPool();
  tp->WaitForIdle();
}

Status FSCoordinator::Sync(const std::string& barrier) {
  Status s = Sink(barrier + "/", std::to_string(server_id_));
  LOG_RETURN_IF_NOT_OK(s)

  while (!IsReady(barrier)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return s;
}

bool FSCoordinator::IsReady(const std::string& barrier) {
  if (IsMaster()) {
    if (Counting(barrier + "/") == server_count_) {
      if (Sink("", barrier + "_done").ok()) {
        LOG(INFO) << "Master sync " << barrier + "_done";
        return true;
      }
    }
    return false;
  } else {
    if (FileExist(barrier + "_done")) {
      LOG(INFO) << "Server " << server_id_ << " monitored "
                << barrier + "_done.";
      return true;
    }
    return false;
  }
}

Status FSCoordinator::Start() {
  return Sink("start/", std::to_string(server_id_));
}

Status FSCoordinator::SetStarted(int32_t server_id) {
  state_ = kStarted;
  return Status::OK();
}

Status FSCoordinator::Init() {
  return Sink("init/", std::to_string(server_id_));
}

Status FSCoordinator::SetInited(int32_t server_id) {
  state_ = kInited;
  return Status::OK();
}

Status FSCoordinator::Prepare() {
  return Sink("prepare/", std::to_string(server_id_));
}

Status FSCoordinator::SetReady(int32_t server_id) {
  state_ = kReady;
  return Status::OK();
}

Status FSCoordinator::Stop(int32_t client_id, int32_t client_count) {
  client_count_ = client_count;
  return Sink("stop/", std::to_string(client_id));
}

Status FSCoordinator::SetStopped(int32_t client_id, int32_t client_count) {
  state_ = kStopped;
  return Status::OK();
}

void FSCoordinator::Refresh() {
  Coordinator::Refresh();
}

void FSCoordinator::CheckStarted() {
  if (IsMaster()) {
    if (Counting("start/") == server_count_) {
      if (Sink("", "started").ok()) {
        SetStarted();
        LOG(INFO) << "Master sync started.";
      }
    }
  } else {
    if (FileExist("started")) {
      SetStarted();
      LOG(INFO) << "Server " << server_id_ << " monitored started.";
    }
  }
}

void FSCoordinator::CheckInited() {
  if (IsMaster()) {
    if (Counting("init/") == server_count_) {
      if (Sink("", "inited").ok()) {
        SetInited();
        LOG(INFO) << "Master sync inited.";
      }
    }
  } else {
    if (FileExist("inited")) {
      SetInited();
      LOG(INFO) << "Server " << server_id_ << " monitored inited.";
    }
  }
}

void FSCoordinator::CheckReady() {
  if (IsMaster()) {
    if (Counting("prepare/") == server_count_) {
      if (Sink("", "ready").ok()) {
        SetReady();
        LOG(INFO) << "Master sync ready.";
      }
    }
  } else {
    if (FileExist("ready")) {
      SetReady();
      LOG(INFO) << "Server " << server_id_ << " monitored ready.";
    }
  }
}

void FSCoordinator::CheckStopped() {
  if (IsMaster()) {
    if (Counting("stop/") == client_count_) {
      if (Sink("", "stopped").ok()) {
        SetStopped();
        LOG(INFO) << "Master sync stopped.";
      }
    }
  } else {
    if (FileExist("stopped")) {
      SetStopped();
      LOG(INFO) << "Server " << server_id_ << " monitored stopped.";
    }
  }
}

bool FSCoordinator::FileExist(const std::string& file_name) {
  // !!! NOTE !!!
  // For some file system, such as NFS, a huge latency exists when using
  // fs_->FileExists(tracker_ + file_name). Instead, we list the directory
  // and then check the files one by one.
  //
  // IF your file system works well, please rewrite this function.
  std::vector<std::string> file_names;
  Status s = fs_->ListDir(tracker_, &file_names);
  if (s.ok()) {
    for (size_t i = 0; i < file_names.size(); ++i) {
      if (file_names[i] == file_name) {
        return true;
      }
    }
  } else {
    LOG(WARNING) << file_name << " check failed: " << s.ToString();
    return false;
  }
  return false;
}

int32_t FSCoordinator::Counting(const std::string& sub_dir) {
  std::vector<std::string> file_names;
  Status s = fs_->ListDir(tracker_ + sub_dir, &file_names);
  if (s.ok()) {
    return file_names.size();
  } else {
    LOG(WARNING) << "Counting states failed: " << sub_dir
                << ", " << s.ToString();
    return 0;
  }
}

Status FSCoordinator::Sink(const std::string& sub_dir,
                           const std::string& file_name) {
  Status s;
  int32_t retry = 0;
  while (retry < GLOBAL_FLAG(RetryTimes)) {
    s = fs_->CreateDir(tracker_ + sub_dir);
    if (s.ok() || error::IsAlreadyExists(s)) {
      LOG(INFO) << "Coordinator sink " << tracker_ << sub_dir;
      break;
    } else {
      LOG(WARNING) << "Coordinator sink " << tracker_ << sub_dir
                  << " failed, try " << retry;
    }
    sleep(1 << retry);
    ++retry;
  }

  retry = 0;
  std::string name = tracker_ + sub_dir + file_name;
  while (retry < GLOBAL_FLAG(RetryTimes)) {
    std::unique_ptr<WritableFile> ret;
    s = fs_->NewWritableFile(name, &ret);
    if (s.ok() || error::IsAlreadyExists(s)) {
      s = ret->Close();
      break;
    } else {
      LOG(WARNING) << "Coordinator sink " << file_name
                  << " failed, try " << retry;
    }
    sleep(1 << retry);
    ++retry;
  }

  LOG(INFO) << "Sink " << name << s.ToString();

  if (error::IsAlreadyExists(s)) {
    return Status::OK();
  } else {
    return s;
  }
}

}  // namespace graphlearn
