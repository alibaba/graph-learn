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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_ALIAS_METHOD_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_ALIAS_METHOD_H_

#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include <vector>
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/core/graph/storage/types.h"

namespace graphlearn {
namespace op {

class AliasMethod {
public:
  AliasMethod();
  explicit AliasMethod(const std::vector<float>* dist);
  explicit AliasMethod(int32_t uniform_max); // uniform distribution AM.

  AliasMethod(const AliasMethod& rhs);
  AliasMethod& operator=(const AliasMethod& rhs);

  bool Sample(int32_t num, int32_t* ret);

private:
  void Build(const std::vector<float>* dist);

private:
  int32_t              range_;
  std::vector<int32_t> alias_;
  std::vector<float>   probs_;
};

class AliasMethodFactory {
public:
  static AliasMethodFactory* GetInstance() {
    static AliasMethodFactory factory;
    return &factory;
  }
  
  template<class T>
  AliasMethod* LookupOrCreate(const std::string& type, 
      const std::vector<T>* weights) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      std::vector<float> tmp_w(weights->begin(), weights->end());
      auto am = new AliasMethod(&tmp_w);
      map_[type] = am;
      return am;
    } else {
      return it->second;
    }                          
  }

  inline AliasMethod* LookupOrCreate(const std::string& type, 
      const std::vector<float>* weights) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      auto am = new AliasMethod(weights);
      map_[type] = am;
      return am;
    } else {
      return it->second;
    }   
  }

  inline AliasMethod* LookupOrCreate(const std::string& type, 
      int32_t uniform_max) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      auto am = new AliasMethod(uniform_max);
      map_[type] = am;
      return am;
    } else {
      return it->second;
    }   
  }

private:
  AliasMethodFactory() {}

private:
  std::mutex mtx_;
  std::unordered_map<std::string, AliasMethod*> map_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_ALIAS_METHOD_H_
