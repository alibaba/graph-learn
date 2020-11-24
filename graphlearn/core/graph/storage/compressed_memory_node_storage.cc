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

#include <mutex>   // NOLINT [build/c++11]
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace io {

class CompressedMemoryNodeStorage : public NodeStorage {
public:
  CompressedMemoryNodeStorage()
      : attributes_(nullptr) {
    int64_t estimate_size = GLOBAL_FLAG(AverageNodeCount);
    id_to_index_.rehash(estimate_size);
    ids_.reserve(estimate_size);
  }

  virtual ~CompressedMemoryNodeStorage() {
    delete attributes_;
  }

  void Lock() override {
    mtx_.lock();
  }

  void Unlock() override {
    mtx_.unlock();
  }

  void SetSideInfo(const SideInfo* info) override {
    if (!side_info_.IsInitialized()) {
      side_info_.CopyFrom(*info);
      if (side_info_.IsAttributed()) {
        attributes_ = NewDataHeldAttributeValue();
      }
    }
  }

  const SideInfo* GetSideInfo() const override {
    return &side_info_;
  }

  void Build() override {
    ids_.shrink_to_fit();
    labels_.shrink_to_fit();
    weights_.shrink_to_fit();
    if (attributes_) {
      attributes_->Shrink();
    }
  }

  IdType Size() const override {
    return ids_.size();
  }

  void Add(NodeValue* value) override {
    if (!Validate(value)) {
      LOG(WARNING) << "Ignore an invalid node value";
      return;
    }

    auto ret = id_to_index_.insert({value->id, ids_.size()});
    if (!ret.second) {
      return;
    }

    ids_.push_back(value->id);

    if (side_info_.IsWeighted()) {
      weights_.push_back(value->weight);
    }
    if (side_info_.IsLabeled()) {
      labels_.push_back(value->label);
    }
    if (side_info_.IsAttributed()) {
      auto ints = value->attrs->GetInts(nullptr);
      for (int32_t i = 0; i < side_info_.i_num; ++i) {
        attributes_->Add(ints[i]);
      }
      auto floats = value->attrs->GetFloats(nullptr);
      for (int32_t i = 0; i < side_info_.f_num; ++i) {
        attributes_->Add(floats[i]);
      }
      auto ss = value->attrs->GetStrings(nullptr);
      for (int32_t i = 0; i < side_info_.s_num; ++i) {
        attributes_->Add(ss[i]);
      }
    }
  }

  float GetWeight(IdType node_id) const override {
    if (!side_info_.IsWeighted()) {
      return 0.0;
    }

    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
      return 0.0;
    } else {
      return weights_[it->second];
    }
  }

  int32_t GetLabel(IdType node_id) const override {
    if (!side_info_.IsLabeled()) {
      return -1;
    }

    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
      return -1;
    } else {
      return labels_[it->second];
    }
  }

  Attribute GetAttribute(IdType node_id) const override {
    if (!side_info_.IsAttributed()) {
      return Attribute();
    }

    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
      return Attribute(AttributeValue::Default(&side_info_), false);
    } else {
      auto value = NewDataRefAttributeValue();
      if (side_info_.i_num > 0) {
        int64_t offset = it->second * side_info_.i_num;
        value->Add(attributes_->GetInts(nullptr) + offset, side_info_.i_num);
      }
      if (side_info_.f_num > 0) {
        int64_t offset = it->second * side_info_.f_num;
        value->Add(attributes_->GetFloats(nullptr) + offset, side_info_.f_num);
      }
      if (side_info_.s_num > 0) {
        int64_t offset = it->second * side_info_.s_num;
        auto ss = attributes_->GetStrings(nullptr) + offset;
        for (int32_t i = 0; i < side_info_.s_num; ++i) {
          value->Add(ss[i].c_str(), ss[i].length());
        }
      }
      return Attribute(value, true);
    }
  }

  const IdArray GetIds() const override {
    return IdArray(ids_.data(), ids_.size());
  }

  const Array<float> GetWeights() const override {
    return Array<float>(weights_);
  }

  const Array<int32_t> GetLabels() const override {
    return Array<int32_t>(labels_);
  }

  const std::vector<Attribute>* GetAttributes() const override {
    return nullptr;
  }

private:
  bool Validate(NodeValue* value) {
    if (!side_info_.IsAttributed()) {
      return true;
    }

    int32_t len = 0;
    value->attrs->GetInts(&len);
    if (len != side_info_.i_num) {
      LOG(WARNING) << "Unmatched int attributes count";
      return false;
    }
    value->attrs->GetFloats(&len);
    if (len != side_info_.f_num) {
      LOG(WARNING) << "Unmatched float attributes count";
      return false;
    }
    value->attrs->GetStrings(&len);
    if (len != side_info_.s_num) {
      LOG(WARNING) << "Unmatched string attributes count";
      return false;
    }
    return true;
  }

private:
  std::mutex mtx_;
  MAP        id_to_index_;
  IdList     ids_;
  std::vector<float>   weights_;
  std::vector<int32_t> labels_;
  AttributeValue*      attributes_;
  SideInfo             side_info_;
};

NodeStorage* NewCompressedMemoryNodeStorage() {
  return new CompressedMemoryNodeStorage();
}

}  // namespace io
}  // namespace graphlearn
