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

#include "graphlearn/common/threading/sync/waitable_event.h"

#include "graphlearn/common/threading/atomic/atomic.h"
#include "graphlearn/common/threading/sync/cond.h"
#include "graphlearn/common/threading/sync/lock.h"

namespace graphlearn {

class WaitableEvent::Impl : private Uncopyable {
public:
  explicit Impl(bool autoReset)
    : cond_(&lock_),
      auto_reset_(autoReset),
      is_set_(false),
      ref_count_(0) {
  }

  ~Impl() {
  }

  bool Wait(int64_t delayInMs = -1) {
    LockType::Locker _(lock_);
    if (!is_set_ && delayInMs != 0) {
      if (!cond_.TimedWait(delayInMs)) {
        return false;
      }
    }
    if (is_set_) {
      if (auto_reset_) {
        is_set_ = false;
      }
      return true;
    }
    return false;
  }

  void Reset() {
    LockType::Locker _(lock_);
    is_set_ = false;
  }

  void Set() {
    LockType::Locker _(lock_);
    is_set_ = true;
    cond_.Signal();
  }

  void Broadcast() {
    LockType::Locker _(lock_);
    cond_.Broadcast();
  }

  void Ref() {
    AtomicIncrement(&ref_count_);
  }

  void Unref() {
    if (AtomicDecrement(&ref_count_) == 0) {
      delete this;
    }
  }

private:
  using LockType = RestrictMutex;
  LockType lock_;
  ConditionVariable cond_;
  bool auto_reset_;
  bool is_set_;
  int32_t ref_count_;
};

WaitableEvent::WaitableEvent(bool autoReset) {
  impl_ = new Impl(autoReset);
  impl_->Ref();
}

WaitableEvent::~WaitableEvent() {
  // Wake all the waiting threads up.
  impl_->Broadcast();
  impl_->Unref();
}

bool WaitableEvent::Wait(int64_t delayInMs) {
  impl_->Ref();
  bool s = impl_->Wait(delayInMs);
  impl_->Unref();
  return s;
}

void WaitableEvent::Reset() {
  impl_->Ref();
  impl_->Reset();
  impl_->Unref();
}

void WaitableEvent::Set() {
  // Use impl_->Unref() is unsafe here! The user may use a WaitableEvent to
  // wait another thread to set an event and then delete same object.  The
  // WaitableEvent may be a member of the object to be deleted.  In this
  // case, access impl_ again will introduce a segment fault.
  Impl* impl = impl_;
  impl->Ref();
  impl->Set();
  impl->Unref();
}

}  // namespace graphlearn
