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

#include "graphlearn/core/operator/aggregator/aggregator.h"

#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/include/config.h"
#include "graphlearn/common/base/log.h"

namespace graphlearn {
namespace op {

Status Aggregator::Aggregate(const AggregateNodesRequest* req,
                             AggregateNodesResponse* res) {
  Noder* node = graph_store_->GetNoder(req->NodeType());
  ::graphlearn::io::NodeStorage* storage = node->GetLocalStorage();
  const ::graphlearn::io::SideInfo* info = storage->GetSideInfo();
  res->SetSideInfo(info, req->NumSegments());

  // default values
  float default_weight = 0.0;
  int32_t default_label = -1;

  // Initialize float attributes.
  // Aggregator only takes effect on float attributes.
  // Weight, label, int attribute and string attribute will be the
  // default value.
  ::graphlearn::io::Attribute attr =
    *(::graphlearn::io::Attribute::Default(info));

  int64_t node_id = 0;
  int32_t segment = 0;
  AggregateNodesRequest* request = const_cast<AggregateNodesRequest*>(req);
  while (request->NextSegment(&segment)) {
    this->InitFunc(&attr.f_attrs, info->f_num);
    for (int32_t i = 0; i < segment; ++i) {
      if (!request->NextId(&node_id)) {
        LOG(WARNING) << "Aggregation: wrong size of segments.";
      }
      this->AggFunc(
        &attr.f_attrs, storage->GetAttribute(node_id)->f_attrs);
    }

    res->AppendWeight(default_weight);
    res->AppendLabel(default_label);

    this->FinalFunc(&attr.f_attrs, segment);
    res->AppendAttribute(&attr);
  }
  return Status::OK();
}

void Aggregator::InitFunc(std::vector<float>* value, int32_t size) {
  value->assign(size, 0.0);
}

void Aggregator::AggFunc(std::vector<float>* left,
                         const std::vector<float>& right) {
}

void Aggregator::FinalFunc(std::vector<float>* values, int32_t total) {
  if (total == 0) {
    values->assign(values->size(), GLOBAL_FLAG(DefaultFloatAttribute));
  }
}

}  // namespace op
}  // namespace graphlearn
