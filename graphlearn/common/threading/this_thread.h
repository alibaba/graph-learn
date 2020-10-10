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

#ifndef GRAPHLEARN_COMMON_THREADING_THIS_THREAD_H_
#define GRAPHLEARN_COMMON_THREADING_THIS_THREAD_H_

#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace graphlearn {

class ThisThread {
public:
  ThisThread() = delete;
  ThisThread(const ThisThread& thread) = delete;
  ThisThread& operator=(const ThisThread& right) = delete;
  ThisThread(ThisThread&& thread) = delete;
  ThisThread& operator=(ThisThread&& right) = delete;

  static void SleepInMs(int ms) {
    ::usleep(ms * 1000);
  }

  static void SleepInUs(int us) {
    ::usleep(us);
  }

  static void Yield() {
#if __APPLE__
    ::sched_yield();
#else
    ::pthread_yield();
#endif
  }

  static int GetId() {
    static __thread pid_t tid = 0;
    if (tid == 0) {
      tid = ::syscall(SYS_gettid);
    }
    return tid;
  }

#if __APPLE__
  static int GetThreadId() {
    uint64_t thread_id;
    ::pthread_threadid_np(::pthread_self(), &thread_id);
    return static_cast<int>(thread_id);
  }
#else
  static pthread_t GetThreadId() {
    return ::pthread_self();
  }
#endif
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_THIS_THREAD_H_
