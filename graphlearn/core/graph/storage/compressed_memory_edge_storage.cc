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
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace io {

class CompressedMemoryEdgeStorage : public EdgeStorage {
public:
  CompressedMemoryEdgeStorage()
      : attributes_(nullptr) {
    int64_t estimate_size = GLOBAL_FLAG(AverageEdgeCount);
    src_ids_.reserve(estimate_size);
    dst_ids_.reserve(estimate_size);
  }

  virtual ~CompressedMemoryEdgeStorage() {
    delete attributes_;
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
    src_ids_.shrink_to_fit();
    dst_ids_.shrink_to_fit();
    labels_.shrink_to_fit();
    weights_.shrink_to_fit();
    if (attributes_) {
      attributes_->Shrink();
    }
  }

  IdType Size() const override {
    return src_ids_.size();
  }

  IdType Add(EdgeValue* value) override {
    if (!Validate(value)) {
      LOG(WARNING) << "Ignore an invalid edge value";
      return -1;
    }

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
    return edge_id;
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
    if (edge_id < Size()) {
      auto value = NewDataRefAttributeValue();
      if (side_info_.i_num > 0) {
        int64_t offset = edge_id * side_info_.i_num;
        value->Add(attributes_->GetInts(nullptr) + offset, side_info_.i_num);
      }
      if (side_info_.f_num > 0) {
        int64_t offset = edge_id * side_info_.f_num;
        value->Add(attributes_->GetFloats(nullptr) + offset, side_info_.f_num);
      }
      if (side_info_.s_num > 0) {
        int64_t offset = edge_id * side_info_.s_num;
        auto ss = attributes_->GetStrings(nullptr) + offset;
        for (int32_t i = 0; i < side_info_.s_num; ++i) {
          value->Add(ss[i].c_str(), ss[i].length());
        }
      }
      return Attribute(value, true);
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

  const std::vector<float>* GetWeights() const override {
    return &weights_;
  }

  const std::vector<int32_t>* GetLabels() const override {
    return &labels_;
  }

  const std::vector<Attribute>* GetAttributes() const override {
    return nullptr;
  }

private:
  bool Validate(EdgeValue* value) {
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
  IdList     src_ids_;
  IdList     dst_ids_;
  std::vector<float>   weights_;
  std::vector<int32_t> labels_;
  AttributeValue*      attributes_;
  SideInfo             side_info_;
};

EdgeStorage* NewCompressedMemoryEdgeStorage() {
  return new CompressedMemoryEdgeStorage();
}

}  // namespace io
}  // namespace graphlearn
