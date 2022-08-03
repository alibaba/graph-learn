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

#include <vector>
#include "core/graph/storage/edge_storage.h"
#include "include/config.h"

namespace graphlearn {
namespace io {

class MemoryEdgeStorage : public EdgeStorage {
public:
  MemoryEdgeStorage() {
    int64_t estimate_size = GLOBAL_FLAG(AverageEdgeCount);
    src_ids_.reserve(estimate_size);
    dst_ids_.reserve(estimate_size);
  }

  virtual ~MemoryEdgeStorage() {
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
    src_ids_.shrink_to_fit();
    dst_ids_.shrink_to_fit();
    labels_.shrink_to_fit();
    weights_.shrink_to_fit();
    attributes_.shrink_to_fit();
  }

  IdType Add(EdgeValue* value) override {
    IdType edge_id = src_ids_.size();

    src_ids_.push_back(value->src_id);
    dst_ids_.push_back(value->dst_id);

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

    return edge_id;
  }

  IdType Size() const override {
    return src_ids_.size();
  }

  IdType GetSrcId(IdType edge_id) const override {
    if (edge_id < Size()) {
      return src_ids_[edge_id];
    } else {
      return -1;
    }
  }

  IdType GetDstId(IdType edge_id) const override {
    if (edge_id < Size()) {
      return dst_ids_[edge_id];
    } else {
      return -1;
    }
  }

  float GetWeight(IdType edge_id) const override {
    if (edge_id < weights_.size()) {
      return weights_[edge_id];
    } else {
      return 0.0;
    }
  }

  int32_t GetLabel(IdType edge_id) const override {
    if (edge_id < labels_.size()) {
      return labels_[edge_id];
    } else {
      return -1;
    }
  }

  Attribute GetAttribute(IdType edge_id) const override {
    if (!side_info_.IsAttributed()) {
      return Attribute();
    }

    if (edge_id < attributes_.size()) {
      return Attribute(attributes_[edge_id].get(), false);
    } else {
      return Attribute(AttributeValue::Default(&side_info_), false);
    }
  }

  const IdArray GetSrcIds() const override {
    return IdArray(src_ids_.data(), src_ids_.size());
  }

  const IdArray GetDstIds() const override {
    return IdArray(dst_ids_.data(), dst_ids_.size());
  }

  const Array<float> GetWeights() const override {
    return Array<float>(weights_);
  }

  const Array<int32_t> GetLabels() const override {
    return Array<int32_t>(labels_);
  }

  const std::vector<Attribute>* GetAttributes() const override {
    return &attributes_;
  }

private:
  IdList     src_ids_;
  IdList     dst_ids_;
  std::vector<int32_t>   labels_;
  std::vector<float>     weights_;
  std::vector<Attribute> attributes_;
  SideInfo               side_info_;
};

EdgeStorage* NewMemoryEdgeStorage() {
  return new MemoryEdgeStorage();
}

}  // namespace io
}  // namespace graphlearn
