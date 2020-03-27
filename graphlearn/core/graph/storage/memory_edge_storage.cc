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
#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/include/config.h"

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
    for (size_t i = 0; i < attributes_.size(); ++i) {
      delete attributes_[i];
    }
  }

  void SetSideInfo(const SideInfo* info) override {
    if (!side_info_.IsInitialized()) {
      side_info_.CopyFrom(*info);
    }
  }

  const SideInfo* GetSideInfo() const override {
    return &side_info_;
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
      Attribute* attr = new Attribute();
      attributes_.push_back(attr);
      attr->i_attrs.swap(value->i_attrs);
      attr->f_attrs.swap(value->f_attrs);
      attr->s_attrs.swap(value->s_attrs);
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

  IndexType GetLabel(IdType edge_id) const override {
    if (edge_id < labels_.size()) {
      return labels_[edge_id];
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

  const Attribute* GetAttribute(IdType edge_id) const override {
    if (edge_id < attributes_.size()) {
      return attributes_[edge_id];
    } else {
      return AttributeValue::Default(&side_info_);
    }
  }

  const IdList* GetSrcIds() const override {
    return &src_ids_;
  }

  const IdList* GetDstIds() const override {
    return &dst_ids_;
  }

  const IndexList* GetLabels() const override {
    return &labels_;
  }

  const std::vector<float>* GetWeights() const override {
    return &weights_;
  }

  const std::vector<Attribute*>* GetAttributes() const override {
    return &attributes_;
  }

private:
  IdList     src_ids_;
  IdList     dst_ids_;
  IndexList  labels_;
  std::vector<float>      weights_;
  std::vector<Attribute*> attributes_;
  SideInfo                side_info_;
};

EdgeStorage* NewMemoryEdgeStorage() {
  return new MemoryEdgeStorage();
}

}  // namespace io
}  // namespace graphlearn
