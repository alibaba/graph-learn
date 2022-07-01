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

#include "graphlearn/common/threading/thread/thread.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include "graphlearn/common/base/uncopyable.h"
#include "graphlearn/common/threading/sync/waitable_event.h"

namespace graphlearn {

namespace {

void* ThreadFunc(void* ctx) {
  ::PB_NAMESPACE::Closure* func =
  reinterpret_cast< ::PB_NAMESPACE::Closure*>(ctx);
  func->Run();
  return nullptr;
}

}  // anonymous namespace

class Thread : private Uncopyable {
private:
  using Callback =  ::PB_NAMESPACE::Closure;
  using handle_type =  pthread_t;

  Thread(Callback* func,
         WaitableEvent* event,
         WaitableEvent* wait,
         const char* name);

  ~Thread();

  void Routine();

private:
  friend ThreadHandle CreateThread(Thread::Callback* func,
                                   WaitableEvent* wait,
                                   const char* name);

  std::string name_;
  Callback* func_;
  WaitableEvent* event_;
  WaitableEvent* wait_;
  handle_type handle_;
};

Thread::Thread(Thread::Callback* func,
               WaitableEvent* event,
               WaitableEvent* wait,
               const char* name)
    : func_(func),
      event_(event),
      wait_(wait),
      handle_(0) {
  if (name != nullptr) {
    name_ = name;
  }
  Callback* routine = NewClosure(this, &Thread::Routine);
  if (::pthread_create(
        &handle_, nullptr, ThreadFunc, reinterpret_cast<void*>(routine))) {
    ::abort();
  }
}

Thread::~Thread() {
  delete event_;
}

void Thread::Routine() {
  func_->Run();
  event_->Wait();
  if (wait_ != nullptr) {
    wait_->Set();
  }
  delete this;
}

ThreadHandle CreateThread(::PB_NAMESPACE::Closure* func,
                          WaitableEvent* wait,
                          const char* name) {
  WaitableEvent* event = new WaitableEvent();
  Thread* thread = new Thread(func, event, wait, name);
  assert(thread);
  Thread::handle_type handle = thread->handle_;
  ::pthread_detach(handle);

  event->Set();

  return handle;
}

}  // namespace graphlearn
