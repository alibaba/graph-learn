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

#ifndef GRAPHLEARN_SERVICE_LOCAL_EVENT_QUEUE_H_
#define GRAPHLEARN_SERVICE_LOCAL_EVENT_QUEUE_H_

#include <unistd.h>
#include <atomic>
#include <mutex>  // NOLINT [build/c++11]
#include "include/config.h"
#include "common/threading/lockfree/lockfree_queue.h"

namespace graphlearn {

template<class T>
class EventQueue {
public:
  explicit EventQueue(size_t capacity)
    : cancel_(false), capacity_(capacity), size_(0) {
    queue_ = new LockFreeQueue<T>();
  }

  ~EventQueue() {
    delete queue_;
  }

  bool Push(const T& t) {
    while (!cancel_) {
      if (size_ < capacity_) {
        ++size_;
        return queue_->Push(t);
      } else {
        usleep(10);
      }
    }
    return false;
  }

  bool Pop(T* t) {
    while (!cancel_) {
      if (queue_->Pop(t)) {
        --size_;
        return true;
      } else {
        usleep(10);
      }
    }
    return false;
  }

  void Cancel() {
    cancel_ = true;
  }

private:
  volatile bool     cancel_;
  size_t            capacity_;
  std::atomic<int>  size_;
  LockFreeQueue<T>* queue_;
};

template<class T>
EventQueue<T>* GetInMemoryEventQueue() {
  static std::mutex mtx;
  static EventQueue<T>* q = NULL;
  if (q == NULL) {
    mtx.lock();
    if (q == NULL) {
      q = new EventQueue<T>(GLOBAL_FLAG(InMemoryQueueSize));
    }
    mtx.unlock();
  }
  return q;
}

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_LOCAL_EVENT_QUEUE_H_
