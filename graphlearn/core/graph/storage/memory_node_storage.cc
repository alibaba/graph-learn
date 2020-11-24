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
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace io {

class MemoryNodeStorage : public NodeStorage {
public:
  MemoryNodeStorage() {
    int64_t estimate_size = GLOBAL_FLAG(AverageNodeCount);

    id_to_index_.rehash(estimate_size);
    ids_.reserve(estimate_size);
  }

  virtual ~MemoryNodeStorage() {
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
    }
  }

  const SideInfo* GetSideInfo() const override {
    return &side_info_;
  }

  void Build() override {
    ids_.shrink_to_fit();
    labels_.shrink_to_fit();
    weights_.shrink_to_fit();
    attributes_.shrink_to_fit();
  }

  IdType Size() const override {
    return ids_.size();
  }

  void Add(NodeValue* value) override {
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
      AttributeValue* attr = NewDataHeldAttributeValue();
      attr->Swap(value->attrs);
      attributes_.emplace_back(attr, true);
    }
  }

  IndexType GetLabel(IdType node_id) const override {
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

  Attribute GetAttribute(IdType node_id) const override {
    if (!side_info_.IsAttributed()) {
      return Attribute();
    }

    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
      return Attribute(AttributeValue::Default(&side_info_), false);
    } else {
      return Attribute(attributes_[it->second].get(), false);
    }
  }

  const IdArray GetIds() const override {
    return IdArray(ids_.data(), ids_.size());
  }

  const std::vector<int32_t>* GetLabels() const override {
    return &labels_;
  }

  const std::vector<float>* GetWeights() const override {
    return &weights_;
  }

  const std::vector<Attribute>* GetAttributes() const override {
    return &attributes_;
  }

private:
  std::mutex mtx_;
  MAP        id_to_index_;
  IdList     ids_;
  std::vector<float>     weights_;
  std::vector<int32_t>   labels_;
  std::vector<Attribute> attributes_;
  SideInfo               side_info_;
};

NodeStorage* NewMemoryNodeStorage() {
  return new MemoryNodeStorage();
}

}  // namespace io
}  // namespace graphlearn
