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

#ifndef GRAPHLEARN_CORE_OPERATOR_SAMPLER_CONDITION_TABLE_H_
#define GRAPHLEARN_CORE_OPERATOR_SAMPLER_CONDITION_TABLE_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "include/sampling_request.h"
#include "common/threading/sync/lock.h"
#include "core/operator/sampler/attribute_nodes_map.h"
#include "core/operator/utils/get_node_attributes_wrapper.h"

namespace graphlearn {
namespace op {

// Selected columns and their proportion for condition sampling.
struct SelectedColumns {
  std::vector<int32_t> int_cols_;
  std::vector<float> int_props_;
  std::vector<int32_t> float_cols_;
  std::vector<float> float_props_;
  std::vector<int32_t> str_cols_;
  std::vector<float> str_props_;

  SelectedColumns() {}
  SelectedColumns(const std::vector<int32_t>& int_cols,
                  const std::vector<float>& int_props,
                  const std::vector<int32_t>& float_cols,
                  const std::vector<float>& float_props,
                  const std::vector<int32_t>& str_cols,
                  const std::vector<float>& str_props) {
    int_cols_ = int_cols;
    int_props_ = int_props;
    float_cols_ = float_cols;
    float_props_ = float_props;
    str_cols_ = str_cols;
    str_props_ = str_props;
  }
};

class ConditionTable {
public:
  explicit ConditionTable(const std::string& id_type,
      const SelectedColumns& selected_cols,
      const std::vector<int64_t>& ids,
      const std::vector<float>& weights);

  explicit ConditionTable(const std::string& id_type,
      const SelectedColumns& selected_cols,
      const std::vector<int64_t>& ids); // no input weight.

  ~ConditionTable();
  
  const Status& GetStatus();
  // Sampling on feature->nodes mapping tables using input
  // int, float and string attributes.
  void Sample(GetNodeAttributesWrapper* attr_wrapper,
              std::unordered_set<int64_t>* nbr_set,
              int32_t neg_num,
              bool unique,
              SamplingResponse* res);

private:
  Status BuildAttrNodesMap(
      const std::vector<int64_t>& ids,
      const std::vector<float>& weights);

  void BatchBuildAttrNodesMap(
      const std::vector<int64_t>& ids,
      const std::vector<float>& weights,
      int32_t start, int32_t end,
      GetNodeAttributesWrapper* attr_wrapper);

private:
  Status status_;
  std::string id_type_;
  SelectedColumns selected_cols_;
  std::vector<AttributeNodesMap<int64_t>> int_attribute_nodes_map_list_;
  std::vector<AttributeNodesMap<float>> float_attribute_nodes_map_list_;
  std::vector<AttributeNodesMap<std::string>> str_attribute_nodes_map_list_;
};

class ConditionTableFactory {
public:
  static ConditionTableFactory* GetInstance() {
    static ConditionTableFactory factory;
    return &factory;
  }

  template<class T>
  ConditionTable* LookupOrCreate(
     const std::string& type,
     const std::string& id_type,
     const SelectedColumns& selected_cols,
     const io::IdArray ids,
     const io::Array<T>& weights) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      std::vector<int64_t> tmp_ids(ids.Size());
      for (size_t idx = 0; idx < ids.Size(); ++idx) {
        tmp_ids[idx] = ids[idx];
      }
      std::vector<float> tmp_w(weights.Size());
      for (size_t idx = 0; idx < weights.Size(); ++idx) {
        tmp_w[idx] = weights[idx];
      }
      auto ct = new ConditionTable(id_type, selected_cols, tmp_ids, tmp_w);
      map_[type] = ct;
      return ct;
    } else {
      return it->second;
    }
  }

  inline ConditionTable* LookupOrCreate(
     const std::string& type,
     const std::string& id_type,
     const SelectedColumns& selected_cols,
     const io::IdArray ids,
     const io::Array<float> weights) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      std::vector<int64_t> tmp_ids(ids.Size());
      for (size_t idx = 0; idx < ids.Size(); ++idx) {
        tmp_ids[idx] = ids[idx];
      }
      std::vector<float> tmp_w(weights.Size());
      for (size_t idx = 0; idx < weights.Size(); ++idx) {
        tmp_w[idx] = weights[idx];
      }
      auto ct = new ConditionTable(id_type, selected_cols, tmp_ids, tmp_w);
      map_[type] = ct;
      return ct;
    } else {
      return it->second;
    }
  }

  inline ConditionTable* LookupOrCreate(
     const std::string& type,
     const std::string& id_type,
     const SelectedColumns& selected_cols,
     const io::IdArray ids) {
    ScopedLocker<std::mutex> _(&mtx_);
    auto it = map_.find(type);
    if (it == map_.end()) {
      std::vector<int64_t> tmp_ids(ids.Size());
      for (size_t idx = 0; idx < ids.Size(); ++idx) {
        tmp_ids[idx] = ids[idx];
      }
      auto ct = new ConditionTable(id_type, selected_cols, tmp_ids);
      map_[type] = ct;
      return ct;
    } else {
      return it->second;
    }
  }

private:
  ConditionTableFactory() {}

private:
  std::mutex mtx_;
  std::unordered_map<std::string, ConditionTable*> map_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_OPERATOR_SAMPLER_CONDITION_TABLE_H_