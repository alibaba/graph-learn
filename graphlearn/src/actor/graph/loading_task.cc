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

#include "actor/graph/loading_task.h"

#include <atomic>
#include <condition_variable> // NOLINT [build/c++11]
#include <string>
#include <utility>
#include "common/base/errors.h"
#include "common/base/log.h"

namespace graphlearn {
namespace actor {

NodeLoadingTask::NodeLoadingTask(io::NodeLoader* reader,
                                 unsigned id,
                                 unsigned batch_size)
    : alien_task(), reader_(reader), source_(nullptr),
      batch_size_(batch_size), output_handle_(id) {
}

void NodeLoadingTask::run() {
  Status s = BeginNextFile();
  while (s.ok()) {
    s = reader_->Read(&value_);
    if (s.ok()) {
      output_handle_.Push(value_, source_wrapper_.GetSideInfo());
    } else if (error::IsOutOfRange(s)) {
      output_handle_.FlushAll();
      s = BeginNextFile();
    } else {
      LOG(ERROR) << "NodeLoadingTask run failed: " << s.ToString();
      break;
    }
    value_.attrs->Clear();
  }

  if (!(s.ok() || error::IsOutOfRange(s))) {
    throw std::runtime_error(s.ToString());
  } else {
    output_handle_.NotifyFinished();
  }
}

EdgeLoadingTask::EdgeLoadingTask(io::EdgeLoader* reader,
                                 unsigned id,
                                 unsigned batch_size)
    : alien_task(), reader_(reader), source_(nullptr),
      batch_size_(batch_size), output_handle_(id) {
}

void EdgeLoadingTask::run() {
  Status s = BeginNextFile();
  while (s.ok()) {
    s = reader_->Read(&value_);
    if (s.ok()) {
      output_handle_.Push(value_, source_wrapper_.GetSideInfo());
      // ignore and go on
    } else if (error::IsOutOfRange(s)) {
      output_handle_.FlushAll();
      s = BeginNextFile();
    } else {
      LOG(ERROR) << "EdgeLoadingTask run failed: " << s.ToString();
      break;
    }
    value_.attrs->Clear();
  }

  if (!(s.ok() || error::IsOutOfRange(s))) {
    throw std::runtime_error(s.ToString());
  } else {
    output_handle_.NotifyFinished();
  }
}

}  // namespace actor
}  // namespace graphlearn
