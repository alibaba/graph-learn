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

#include "graphlearn/include/graph_request.h"

#include "graphlearn/core/io/element_value.h"
#include "graphlearn/include/constants.h"
#include "graphlearn/proto/service.pb.h"

namespace graphlearn {

UpdateRequest::UpdateRequest()
    : OpRequest(),
      info_(nullptr),
      cursor_(0) {
}

UpdateRequest::~UpdateRequest() {
  if (is_parse_from_) {
    delete info_;
  }
}

UpdateRequest::UpdateRequest(const io::SideInfo* info, int32_t batch_size)
    : OpRequest(),
      info_(const_cast<io::SideInfo*>(info)),
      cursor_(0) {
  ADD_TENSOR(params_, kSideInfo, kInt32, 4);
  infos_ = &(params_[kSideInfo]);
  infos_->AddInt32(info_->format);
  infos_->AddInt32(info_->i_num);
  infos_->AddInt32(info_->f_num);
  infos_->AddInt32(info_->s_num);

  if (info_->IsWeighted()) {
    ADD_TENSOR(tensors_, kWeightKey, kFloat, batch_size);
    weights_ = &(tensors_[kWeightKey]);
  }
  if (info_->IsLabeled()) {
    ADD_TENSOR(tensors_, kLabelKey, kInt32, batch_size);
    labels_ = &(tensors_[kLabelKey]);
  }
  if (info_->i_num > 0) {
    ADD_TENSOR(tensors_, kIntAttrKey, kInt64, batch_size * info_->i_num);
    i_attrs_ = &(tensors_[kIntAttrKey]);
  }
  if (info_->f_num > 0) {
    ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, batch_size * info_->f_num);
    f_attrs_ = &(tensors_[kFloatAttrKey]);
  }
  if (info_->s_num > 0) {
    ADD_TENSOR(tensors_, kStringAttrKey, kString, batch_size * info_->s_num);
    s_attrs_ = &(tensors_[kStringAttrKey]);
  }
}

void UpdateRequest::SetMembers() {
  infos_ = &(params_[kSideInfo]);

  info_ = new io::SideInfo();
  info_->format = infos_->GetInt32(0);
  info_->i_num = infos_->GetInt32(1);
  info_->f_num = infos_->GetInt32(2);
  info_->s_num = infos_->GetInt32(3);

  if (info_->IsWeighted()) {
    weights_ = &(tensors_[kWeightKey]);
  }
  if (info_->IsLabeled()) {
    labels_ = &(tensors_[kLabelKey]);
  }
  if (info_->i_num > 0) {
    i_attrs_ = &(tensors_[kIntAttrKey]);
  }
  if (info_->f_num > 0) {
    f_attrs_ = &(tensors_[kFloatAttrKey]);
  }
  if (info_->s_num > 0) {
    s_attrs_ = &(tensors_[kStringAttrKey]);
  }
}

void UpdateRequest::Append(const io::AttributeValue* value) {
  if (info_->IsAttributed()) {
    auto ints = value->GetInts(nullptr);
    for (int32_t i = 0; i < info_->i_num; ++i) {
      i_attrs_->AddInt64(ints[i]);
    }

    auto floats = value->GetFloats(nullptr);
    for (int32_t i = 0; i < info_->f_num; ++i) {
      f_attrs_->AddFloat(floats[i]);
    }

    auto ss = value->GetStrings(nullptr);
    for (int32_t i = 0; i < info_->s_num; ++i) {
      s_attrs_->AddString(ss[i]);
    }
  }
}

const io::SideInfo* UpdateRequest::GetSideInfo() const {
  return static_cast<const io::SideInfo*>(info_);
}

void UpdateRequest::Next(io::AttributeValue* value) {
  if (info_->IsAttributed()) {
    value->Clear();

    int32_t from = info_->i_num * cursor_;
    int32_t to = info_->i_num * (cursor_ + 1);
    for (int32_t i = from; i < to; ++i) {
      value->Add(i_attrs_->GetInt64(i));
    }

    from = info_->f_num * cursor_;
    to = info_->f_num * (cursor_ + 1);
    for (int32_t i = from; i < to; ++i) {
      value->Add(f_attrs_->GetFloat(i));
    }

    from = info_->s_num * cursor_;
    to = info_->s_num * (cursor_ + 1);
    for (int32_t i = from; i < to; ++i) {
      value->Add(s_attrs_->GetString(i));
    }
  }
}

UpdateEdgesRequest::UpdateEdgesRequest() : UpdateRequest() {
}

UpdateEdgesRequest::UpdateEdgesRequest(const io::SideInfo* info,
                                       int32_t batch_size)
    : UpdateRequest(info, batch_size) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("UpdateEdges");

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);

  ADD_TENSOR(params_, kEdgeType, kString, 3);
  params_[kEdgeType].AddString(info_->type);
  params_[kEdgeType].AddString(info_->src_type);
  params_[kEdgeType].AddString(info_->dst_type);

  ADD_TENSOR(params_, kDirection, kInt32, 1);
  params_[kDirection].AddInt32(info_->direction);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, batch_size);
  src_ids_ = &(tensors_[kSrcIds]);

  ADD_TENSOR(tensors_, kDstIds, kInt64, batch_size);
  dst_ids_ = &(tensors_[kDstIds]);
}

OpRequest* UpdateEdgesRequest::Clone() const {
  return new UpdateEdgesRequest(info_, Size());
}

void UpdateEdgesRequest::SerializeTo(void* request) {
  OpRequest::SerializeTo(request);
  OpRequestPb* pb = static_cast<OpRequestPb*>(request);
  pb->set_need_server_ready(false);
}

void UpdateEdgesRequest::SetMembers() {
  UpdateRequest::SetMembers();
  info_->type = params_[kEdgeType].GetString(0);
  info_->src_type = params_[kEdgeType].GetString(1);
  info_->dst_type = params_[kEdgeType].GetString(2);
  src_ids_ = &(tensors_[kSrcIds]);
  dst_ids_ = &(tensors_[kDstIds]);
}

int32_t UpdateEdgesRequest::Size() const {
  return src_ids_->Size();
}

void UpdateEdgesRequest::Append(const io::EdgeValue* value) {
  src_ids_->AddInt64(value->src_id);
  dst_ids_->AddInt64(value->dst_id);
  if (info_->IsWeighted()) {
    weights_->AddFloat(value->weight);
  }
  if (info_->IsLabeled()) {
    labels_->AddInt32(value->label);
  }
  UpdateRequest::Append(value->attrs);
}

bool UpdateEdgesRequest::Next(io::EdgeValue* value) {
  if (cursor_ >= Size()) {
    return false;
  }

  value->src_id = src_ids_->GetInt64(cursor_);
  value->dst_id = dst_ids_->GetInt64(cursor_);
  if (info_->IsWeighted()) {
    value->weight = weights_->GetFloat(cursor_);
  }
  if (info_->IsLabeled()) {
    value->label = labels_->GetInt32(cursor_);
  }
  UpdateRequest::Next(value->attrs);
  ++cursor_;
  return true;
}

UpdateNodesRequest::UpdateNodesRequest() : UpdateRequest() {
}

UpdateNodesRequest::UpdateNodesRequest(const io::SideInfo* info,
                                       int32_t batch_size)
    : UpdateRequest(info, batch_size) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("UpdateNodes");

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kNodeIds);

  ADD_TENSOR(params_, kNodeType, kString, 1);
  params_[kNodeType].AddString(info_->type);

  ADD_TENSOR(tensors_, kNodeIds, kInt64, batch_size);
  ids_ = &(tensors_[kNodeIds]);
}

OpRequest* UpdateNodesRequest::Clone() const {
  return new UpdateNodesRequest(info_, Size());
}

void UpdateNodesRequest::SerializeTo(void* request) {
  OpRequest::SerializeTo(request);
  OpRequestPb* pb = static_cast<OpRequestPb*>(request);
  pb->set_need_server_ready(false);
}

void UpdateNodesRequest::SetMembers() {
  UpdateRequest::SetMembers();
  info_->type = params_[kNodeType].GetString(0);
  ids_ = &(tensors_[kNodeIds]);
}

int32_t UpdateNodesRequest::Size() const {
  return ids_->Size();
}

void UpdateNodesRequest::Append(const io::NodeValue* value) {
  ids_->AddInt64(value->id);
  if (info_->IsWeighted()) {
    weights_->AddFloat(value->weight);
  }
  if (info_->IsLabeled()) {
    labels_->AddInt32(value->label);
  }
  UpdateRequest::Append(value->attrs);
}

bool UpdateNodesRequest::Next(io::NodeValue* value) {
  if (cursor_ >= Size()) {
    return false;
  }

  value->id = ids_->GetInt64(cursor_);
  if (info_->IsWeighted()) {
    value->weight = weights_->GetFloat(cursor_);
  }
  if (info_->IsLabeled()) {
    value->label = labels_->GetInt32(cursor_);
  }
  UpdateRequest::Next(value->attrs);
  ++cursor_;
  return true;
}

REGISTER_REQUEST(UpdateEdges, UpdateEdgesRequest, UpdateEdgesResponse);
REGISTER_REQUEST(UpdateNodes, UpdateNodesRequest, UpdateNodesResponse);

}  // namespace graphlearn
