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

#include "graphlearn/common/threading/sync/cond.h"

#include <sys/time.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <thread>  // NOLINT [build/c++11]

namespace graphlearn {

namespace {

const int64_t kMillisecondsPerSecond = 1000;
const int64_t kNanosecondsPerSecond = 1000000000;
const int64_t kNanosecondsPerMillisecond = 1000000;
const int64_t kNanosecondsPerMicrosecond = 1000;

}  // anonymous namespace

ConditionVariable::ConditionVariable(MutexBase* lock)
    : lock_(lock->NativeLock()) {
  ::pthread_cond_init(&cond_, NULL);
}

ConditionVariable::~ConditionVariable() {
  ::pthread_cond_destroy(&cond_);
}

void ConditionVariable::Wait() {
  ::pthread_cond_wait(&cond_, lock_);
}

bool ConditionVariable::TimedWait(int64_t delayInMs) {
  if (delayInMs >= 0) {
    struct timeval now;
    ::gettimeofday(&now, NULL);

    struct timespec ts;
    ts.tv_sec = now.tv_sec + (delayInMs / kMillisecondsPerSecond);
    ts.tv_nsec = now.tv_usec * kNanosecondsPerMicrosecond +
          (delayInMs % kMillisecondsPerSecond)
            * kNanosecondsPerMillisecond;
    ts.tv_sec += ts.tv_nsec / kNanosecondsPerSecond;
    ts.tv_nsec %= kNanosecondsPerSecond;

    int ret = ::pthread_cond_timedwait(&cond_, lock_, &ts);
    if (ret == 0) {
      return true;
    } else if (ret == ETIMEDOUT) {
      return false;
    } else {
      ::abort();
    }
  } else if (delayInMs == -1) {
    Wait();
    return true;
  }

  return false;
}

void ConditionVariable::Signal() {
  ::pthread_cond_signal(&cond_);
}

void ConditionVariable::Broadcast() {
  ::pthread_cond_broadcast(&cond_);
}

SyncVariable::SyncVariable(int32_t th)
  : cond_(0), threshold_(th) {
}

SyncVariable::~SyncVariable() {
}

void SyncVariable::Inc() {
  cond_.fetch_add(1);
}

void SyncVariable::Wait() {
  while (cond_.load() < threshold_) {
    std::this_thread::yield();
  }
}

}  // namespace graphlearn

