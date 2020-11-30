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

#ifndef GRAPHLEARN_CORE_GRAPH_HETER_DISPATCHER_H_
#define GRAPHLEARN_CORE_GRAPH_HETER_DISPATCHER_H_

#include <mutex>  //NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include "graphlearn/common/threading/sync/lock.h"

namespace graphlearn {

template <class T>
class HeterDispatcher {
public:
  typedef T* (*TypeCreator)(const std::string& type,
                            const std::string& view_type,
                            const std::string& use_attrs);

public:
  explicit HeterDispatcher(TypeCreator creator)
      : creator_(creator) {
  }

  ~HeterDispatcher() {
    for (const auto& item : holder_) {
      delete item.second;
    }
  }

  T* LookupOrCreate(const std::string& type,
                    const std::string& view_type="",
                    const std::string& use_attrs="") {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = holder_.find(type);
    if (it != holder_.end()) {
      return it->second;
    }

    T* t = creator_(type, view_type, use_attrs);
    holder_[type] = t;
    return t;
  }

  void ResetNext() {
    it_ = holder_.begin();
  }

  bool Next(std::string* type, T** t) {
    if (it_ == holder_.end()) {
      return false;
    }

    *type = it_->first;
    *t = it_->second;
    ++it_;
    return true;
  }

private:
  TypeCreator creator_;
  std::mutex  mtx_;
  std::unordered_map<std::string, T*> holder_;
  typename std::unordered_map<std::string, T*>::iterator it_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_HETER_DISPATCHER_H_
