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

#include "core/operator/sampler/condition_table.h"

#include "common/base/macros.h"

namespace graphlearn {
namespace op {

namespace {
const int32_t ID_SIZE_PER_REQ = 102400;
}  // anonymous namespace

ConditionTable::ConditionTable(const std::string& id_type,
    const SelectedColumns& selected_cols,
    const std::vector<int64_t>& ids,
    const std::vector<float>& weights) {
  id_type_ = id_type;
  selected_cols_ = selected_cols;
  int_attribute_nodes_map_list_.resize(selected_cols_.int_cols_.size());
  float_attribute_nodes_map_list_.resize(selected_cols_.float_cols_.size());
  str_attribute_nodes_map_list_.resize(selected_cols.str_cols_.size());
  status_ = BuildAttrNodesMap(ids, weights);
}

ConditionTable::ConditionTable(const std::string& id_type,
    const SelectedColumns& selected_cols,
    const std::vector<int64_t>& ids) {
  id_type_ = id_type;
  selected_cols_ = selected_cols;
  int_attribute_nodes_map_list_.resize(selected_cols_.int_cols_.size());
  float_attribute_nodes_map_list_.resize(selected_cols_.float_cols_.size());
  str_attribute_nodes_map_list_.resize(selected_cols.str_cols_.size());
  std::vector<float> weights;
  status_ = BuildAttrNodesMap(ids, weights);
}

ConditionTable::~ConditionTable() {
}

const Status& ConditionTable::GetStatus() {
  return status_;
}

Status ConditionTable::BuildAttrNodesMap(
    const std::vector<int64_t>& ids,
    const std::vector<float>& weights) {
  int32_t remain_size = ids.size();
  int32_t offset = 0;
  int64_t* ids_raw_data = const_cast<int64_t*>(ids.data());
  while(remain_size > ID_SIZE_PER_REQ) {
    // get nodes' attribute.
    GetNodeAttributesWrapper attr_wrapper(
        id_type_, ids_raw_data + offset, ID_SIZE_PER_REQ);
    RETURN_IF_NOT_OK(attr_wrapper.GetStatus())
    // build attribute->nodes tables.
    BatchBuildAttrNodesMap(ids, weights,
        offset, offset + ID_SIZE_PER_REQ, &attr_wrapper);
    offset += ID_SIZE_PER_REQ;
    remain_size -= ID_SIZE_PER_REQ;
  }

  // remain ids
  GetNodeAttributesWrapper attr_wrapper(
      id_type_, ids_raw_data + offset, remain_size);
  RETURN_IF_NOT_OK(attr_wrapper.GetStatus())
  BatchBuildAttrNodesMap(ids, weights,
      offset, offset + remain_size, &attr_wrapper);
  // create alias table for each attribute_nodes map.
  for (auto& item : int_attribute_nodes_map_list_) {
    item.CreateAM();
  }
  for (auto& item : float_attribute_nodes_map_list_) {
    item.CreateAM();
  }
  for (auto& item : str_attribute_nodes_map_list_) {
    item.CreateAM();
  }
  return Status::OK();
}

void ConditionTable::BatchBuildAttrNodesMap(
    const std::vector<int64_t>& ids,
    const std::vector<float>& weights,
    int32_t start, int32_t end,
    GetNodeAttributesWrapper* attr_wrapper) {
  for (int32_t i = start; i < end; i++) {
    int64_t id = ids.at(i);
    float weight = (weights.empty()) ? 1.0 : weights.at(i);
    const int64_t* int_attrs = attr_wrapper->NextIntAttrs();
    for (int32_t j = 0; j < selected_cols_.int_cols_.size(); j++) {
      int_attribute_nodes_map_list_[j].Insert(
        int_attrs[selected_cols_.int_cols_[j]], id, weight);
    }

    const float* float_attrs = attr_wrapper->NextFloatAttrs();
    for (int32_t j = 0; j < selected_cols_.float_cols_.size(); j++) {
      float_attribute_nodes_map_list_[j].Insert(
        float_attrs[selected_cols_.float_cols_[j]], id, weight);
    }

    const std::string* const* str_attrs = attr_wrapper->NextStrAttrs();
    for (int32_t j = 0; j < selected_cols_.str_cols_.size(); j++) {
      str_attribute_nodes_map_list_[j].Insert(
        *(str_attrs[selected_cols_.str_cols_[j]]), id, weight);
    }
  }
}

void ConditionTable::Sample(GetNodeAttributesWrapper* attr_wrapper,
    std::unordered_set<int64_t>* nbr_set,
    int32_t neg_num,
    bool unique,
    SamplingResponse* res) {
#define TYPE_SAMPLE(type, cols, props, attrs,                    \
    nbr_set, neg_num, unique, res)                               \
  for (int32_t i = 0; i < cols.size(); i++) {                    \
    type##_attribute_nodes_map_list_[i].Sample(attrs[cols[i]],   \
        nbr_set, neg_num * props[i], unique, res);               \
  }

  const int64_t* int_attrs = attr_wrapper->NextIntAttrs();
  const float* float_attrs = attr_wrapper->NextFloatAttrs();
  TYPE_SAMPLE(int, selected_cols_.int_cols_, selected_cols_.int_props_,
      int_attrs, nbr_set, neg_num, unique, res)
  TYPE_SAMPLE(float, selected_cols_.float_cols_, selected_cols_.float_props_,
      float_attrs, nbr_set, neg_num, unique, res)
#undef TYPE_SAMPLE
  const std::string* const* str_attrs = attr_wrapper->NextStrAttrs();
  for (int32_t i = 0; i < selected_cols_.str_cols_.size(); i++) {
    str_attribute_nodes_map_list_[i].Sample(
        *(str_attrs[selected_cols_.str_cols_[i]]),
        nbr_set, neg_num * selected_cols_.str_props_[i], unique, res);
  }
}


}  // namespace op
}  // namespace graphlearn
