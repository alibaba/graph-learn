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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_ATTRIBUTE_NODES_MAP_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_ATTRIBUTE_NODES_MAP_H_

#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/core/operator/sampler/alias_method.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/sampling_request.h"


namespace graphlearn {
namespace op {

template<class T>
std::string ToString(const T& item) {
  std::stringstream ss;
  ss << item;
  return ss.str();
};

struct IdWeight{
  IdWeight() {}
  IdWeight(std::vector<int64_t>&& ids, std::vector<float>&& weights):
    ids_(ids), weights_(weights) {
  }
  std::vector<int64_t> ids_;
  std::vector<float> weights_;
};

template<class AttrType>
class AttributeNodesMap {
public:
  ~AttributeNodesMap() {
    for (auto& iter : attr_am_) {
      delete iter.second;
    }
  }
  void Insert(const AttrType& attr, int64_t id, float weight);
  void CreateAM();
  void Sample(const AttrType& attr,
              std::unordered_set<int64_t>* nbr_set,
              int32_t num,
              bool unique,
              SamplingResponse* res);
private:
  std::unordered_map<AttrType, IdWeight> attr_id_weights_;
  std::unordered_map<AttrType, AliasMethod*> attr_am_;
};

template<class AttrType>
void AttributeNodesMap<AttrType>::Insert(
    const AttrType& attr, int64_t id, float weight) {
  auto iter = attr_id_weights_.find(attr);
  if (iter == attr_id_weights_.end()) {
    attr_id_weights_.emplace(attr,
        IdWeight(std::move(std::vector<int64_t>(1, id)),
                 std::move(std::vector<float>(1, weight))));
  } else {
    iter->second.ids_.emplace_back(id);
    iter->second.weights_.emplace_back(weight);
  }
}

template<class AttrType>
void AttributeNodesMap<AttrType>::CreateAM() {
  for (const auto& item : attr_id_weights_) {
    const std::string& name = ToString<>(item.first);
    AliasMethod* am = new AliasMethod(&(item.second.weights_));
    if (attr_am_.find(item.first) == attr_am_.end()) {
      attr_am_.emplace(item.first, am);
    }
  }
}

template<class AttrType>
void AttributeNodesMap<AttrType>::Sample(
    const AttrType& attr,
    std::unordered_set<int64_t>* nbr_set,
    int32_t num,
    bool unique,
    SamplingResponse* res) {
  std::unique_ptr<int32_t[]> indices(new int32_t[num]);
  int32_t retry_times = GLOBAL_FLAG(NegativeSamplingRetryTimes);
  auto iter = attr_am_.find(attr);
  // when there is no this attr at all, just skip
  if (iter == attr_am_.end()) return;
  AliasMethod* am = iter->second;
  int32_t count = 0;
  int32_t cursor = 0;
  while (count < num && retry_times > 0) {
    cursor %= num;
    if (cursor == 0) {
      am->Sample(num, indices.get());
      --retry_times;
    }
    int64_t item = attr_id_weights_[iter->first].ids_.at(indices[cursor++]);
    if (nbr_set->find(item) == nbr_set->end()) {
      res->AppendNeighborId(item);
      ++count;
      if (unique) {
        nbr_set->insert(item);
      }
    }
  }
}

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_ATTRIBUTE_NODES_MAP_H_
