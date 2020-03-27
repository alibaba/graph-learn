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

#include "graphlearn/service/dist/naming_engine.h"

#include <unistd.h>
#include <memory>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/macros.h"
#include "graphlearn/common/string/numeric.h"
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

namespace {
const char* kDone = "done";
}  // anonymous namespace

NamingEngine* NamingEngine::GetInstance() {
  static NamingEngine engine;
  return &engine;
}

NamingEngine::NamingEngine()
    : stopped_(false), size_(0), fs_(nullptr) {
  if (strings::EndWith(GLOBAL_FLAG(Tracker), "/")) {
    tracker_ = GLOBAL_FLAG(Tracker) + "endpoints/";
  } else {
    tracker_ = GLOBAL_FLAG(Tracker) + "/endpoints/";
  }

  Status s = Env::Default()->GetFileSystem(tracker_, &fs_);
  if (!s.ok()) {
    LOG(FATAL) << "Invalid tracker path: " << tracker_;
    ::exit(-1);
  }

  s = fs_->CreateDir(tracker_);
  if (s.ok() || error::IsAlreadyExists(s)) {
    LOG(INFO) << "Connect naming engine ok: " << tracker_;
  } else {
    LOG(FATAL) << "Connect naming engine failed: " << tracker_;
    ::exit(-1);
  }

  endpoints_.resize(GLOBAL_FLAG(ServerCount));

  auto tp = Env::Default()->ReservedThreadPool();
  tp->AddTask(NewClosure(this, &NamingEngine::Refresh));
}

NamingEngine::~NamingEngine() {
  if (!stopped_) {
    Stop();
  }
}

void NamingEngine::Stop() {
  stopped_ = true;
  sleep(1);
}

void NamingEngine::SetCapacity(int32_t capacity) {
  ScopedLocker<std::mutex> _(&mtx_);
  endpoints_.resize(capacity);
}

int32_t NamingEngine::Size() const {
  return size_;
}

std::string NamingEngine::Get(int32_t server_id) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (server_id < endpoints_.size()) {
    return endpoints_.at(server_id);
  } else {
    return "";
  }
}

Status NamingEngine::Update(int32_t server_id,
                            const std::string& endpoint) {
  std::string file_name = tracker_ + std::to_string(server_id);
  LOG(INFO) << "Update endpoint id: " << server_id
            << ", address: " << endpoint
            << ", filepath: " << file_name;

  std::unique_ptr<WritableFile> ret;
  Status s = fs_->NewWritableFile(file_name, &ret);
  RETURN_IF_NOT_OK(s)

  s = ret->Append(endpoint);
  RETURN_IF_NOT_OK(s)

  s = ret->Close();
  RETURN_IF_NOT_OK(s)
  return s;
}

void NamingEngine::Refresh() {
  while (!stopped_) {
    std::vector<std::string> file_names;
    Status s = fs_->ListDir(tracker_, &file_names);
    if (s.ok()) {
      Parse(file_names);
    } else {
      LOG(WARNING) << "Refresh endpoints failed: " << s.ToString();
    }
    sleep(1);
  }
}

void NamingEngine::Parse(const std::vector<std::string>& names) {
  char buffer[32] = {0};

  int32_t count = 0;
  std::vector<std::string> tmp(endpoints_.size(), "");
  for (size_t i = 0; i < names.size(); ++i) {
    int32_t id = -1;
    if (!strings::SafeStringTo32(names[i], &id) ||
        id < 0 || id >= tmp.size()) {
      continue;
    }

    std::unique_ptr<ByteStreamAccessFile> ret;
    Status s = fs_->NewByteStreamAccessFile(tracker_ + names[i], 0, &ret);
    if (!s.ok()) {
      LOG(WARNING) << "Invalid endpoint file: " << names[i];
      continue;
    }

    LiteString endpoint;
    s = ret->Read(sizeof(buffer), &endpoint, buffer);
    if (s.ok()) {
      tmp[id] = endpoint.ToString();
      ++count;
    } else {
      LOG(WARNING) << "Invalid endpoint file: " << names[i];
    }
  }

  ScopedLocker<std::mutex> _(&mtx_);
  LOG(INFO) << "Refresh endpoints count: " << size_;
  size_ = count;
  endpoints_.swap(tmp);
}

}  // namespace graphlearn
