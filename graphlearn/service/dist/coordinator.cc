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

#include "graphlearn/service/dist/coordinator.h"

#include <unistd.h>
#include <memory>
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

Coordinator::Coordinator(int32_t server_id, int32_t server_count,
                         Env* env)
    : started_(false),
      ready_(false),
      stopped_(false),
      client_count_(-1),
      server_id_(server_id),
      server_count_(server_count) {
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
  tp->AddTask(NewClosure(this, &Coordinator::Refresh));
}

Coordinator::~Coordinator() {
  stopped_ = true;
}

bool Coordinator::IsMaster() const {
  return server_id_ == 0;
}

Status Coordinator::Start() {
  return Sink("start/", std::to_string(server_id_));
}

bool Coordinator::IsStartup() const {
  return started_;
}

Status Coordinator::SetReady() {
  return Sink("prepare/", std::to_string(server_id_));
}

bool Coordinator::IsReady() const {
  return ready_;
}

Status Coordinator::Stop(int32_t client_id, int32_t client_count) {
  client_count_ = client_count;
  return Sink("stop/", std::to_string(client_id));
}

bool Coordinator::IsStopped() const {
  return stopped_;
}

void Coordinator::Refresh() {
  while (!stopped_) {
    if (!started_) {
      CheckStarted();
    }
    if (!ready_) {
      CheckReady();
    }
    if (!stopped_) {
      CheckStopped();
    }
    sleep(1);
  }
}

void Coordinator::CheckStarted() {
  if (IsMaster()) {
    if (Counting("start/") == server_count_) {
      if (Sink("", "started").ok()) {
        started_ = true;
        LOG(INFO) << "Master sync started.";
      }
    }
  } else {
    if (FileExist("started")) {
      started_ = true;
      LOG(INFO) << "Server " << server_id_ << " monitored started.";
    }
  }
}

void Coordinator::CheckReady() {
  if (IsMaster()) {
    if (Counting("prepare/") == server_count_) {
      if (Sink("", "ready").ok()) {
        ready_ = true;
        LOG(INFO) << "Master sync ready.";
      }
    }
  } else {
    if (FileExist("ready")) {
      ready_ = true;
      LOG(INFO) << "Server " << server_id_ << " monitored ready.";
    }
  }
}

void Coordinator::CheckStopped() {
  if (IsMaster()) {
    if (Counting("stop/") == client_count_) {
      if (Sink("", "stopped").ok()) {
        stopped_ = true;
        LOG(INFO) << "Master sync stopped.";
      }
    }
  } else {
    if (FileExist("stopped")) {
      stopped_ = true;
      LOG(INFO) << "Server " << server_id_ << " monitored stopped.";
    }
  }
}

bool Coordinator::FileExist(const std::string& file_name) {
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

int32_t Coordinator::Counting(const std::string& sub_dir) {
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

Status Coordinator::Sink(const std::string& sub_dir,
                         const std::string& file_name) {
  Status s;
  int32_t retry = 0;
  while (retry < GLOBAL_FLAG(RetryTimes)) {
    s = fs_->CreateDir(tracker_ + sub_dir);
    if (s.ok() || error::IsAlreadyExists(s)) {
      LOG(INFO) << "Coordinator sink " << sub_dir;
      break;
    } else {
      LOG(WARNING) << "Coordinator sink " << sub_dir
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
      LOG(WARNING) << "Coordinator sink " << sub_dir << file_name
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
