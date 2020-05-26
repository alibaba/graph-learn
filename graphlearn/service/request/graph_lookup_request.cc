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

namespace graphlearn {

namespace {
int32_t kReservedSize = 64;
}  // anonymous namespace

GetEdgesRequest::GetEdgesRequest() : OpRequest() {
}

GetEdgesRequest::GetEdgesRequest(const std::string& edge_type,
                                 const std::string& strategy,
                                 int32_t batch_size,
                                 int32_t epoch)
    : OpRequest() {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("GetEdges");

  ADD_TENSOR(params_, kEdgeType, kString, 2);
  params_[kEdgeType].AddString(edge_type);
  params_[kEdgeType].AddString(strategy);

  ADD_TENSOR(params_, kBatchSize, kInt32, 1);
  params_[kBatchSize].AddInt32(batch_size);

  ADD_TENSOR(params_, kSideInfo, kInt32, 1);
  params_[kSideInfo].AddInt32(epoch);
}

const std::string& GetEdgesRequest::EdgeType() const {
  return params_.at(kEdgeType).GetString(0);
}

const std::string& GetEdgesRequest::Strategy() const {
  return params_.at(kEdgeType).GetString(1);
}

int32_t GetEdgesRequest::BatchSize() const {
  return params_.at(kBatchSize).GetInt32(0);
}

int32_t GetEdgesRequest::Epoch() const {
  return params_.at(kSideInfo).GetInt32(0);
}

GetEdgesResponse::GetEdgesResponse() : OpResponse() {}

void GetEdgesResponse::SetMembers() {
  src_ids_ = &(tensors_[kSrcIds]);
  dst_ids_ = &(tensors_[kDstIds]);
  edge_ids_ = &(tensors_[kEdgeIds]);
}

void GetEdgesResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  GetEdgesResponse& res = static_cast<GetEdgesResponse&>(right);
  std::swap(src_ids_, res.src_ids_);
  std::swap(dst_ids_, res.dst_ids_);
  std::swap(edge_ids_, res.edge_ids_);
}

void GetEdgesResponse::Init(int32_t batch_size) {
  ADD_TENSOR(tensors_, kSrcIds, kInt64, batch_size);
  src_ids_ = &(tensors_[kSrcIds]);

  ADD_TENSOR(tensors_, kDstIds, kInt64, batch_size);
  dst_ids_ = &(tensors_[kDstIds]);

  ADD_TENSOR(tensors_, kEdgeIds, kInt64, batch_size);
  edge_ids_ = &(tensors_[kEdgeIds]);
}

void GetEdgesResponse::Append(int64_t src_id, int64_t dst_id, int64_t edge_id) {
  src_ids_->AddInt64(src_id);
  dst_ids_->AddInt64(dst_id);
  edge_ids_->AddInt64(edge_id);
  ++batch_size_;
}

const int64_t* GetEdgesResponse::SrcIds() const {
  return src_ids_->GetInt64();
}

const int64_t* GetEdgesResponse::DstIds() const {
  return dst_ids_->GetInt64();
}

const int64_t* GetEdgesResponse::EdgeIds() const {
  return edge_ids_->GetInt64();
}

GetNodesRequest::GetNodesRequest() : OpRequest() {
}

GetNodesRequest::GetNodesRequest(const std::string& type,
                                 const std::string& strategy,
                                 NodeFrom node_from,
                                 int32_t batch_size,
                                 int32_t epoch)
    : OpRequest() {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("GetNodes");

  ADD_TENSOR(params_, kNodeType, kString, 2);
  params_[kNodeType].AddString(type);
  params_[kNodeType].AddString(strategy);

  ADD_TENSOR(params_, kSideInfo, kInt32, 3);
  params_[kSideInfo].AddInt32(node_from);
  params_[kSideInfo].AddInt32(batch_size);
  params_[kSideInfo].AddInt32(epoch);
}

const std::string& GetNodesRequest::Type() const {
  return params_.at(kNodeType).GetString(0);
}

const std::string& GetNodesRequest::Strategy() const {
  return params_.at(kNodeType).GetString(1);
}

NodeFrom GetNodesRequest::GetNodeFrom() const {
  return static_cast<NodeFrom>(params_.at(kSideInfo).GetInt32(0));
}

int32_t GetNodesRequest::BatchSize() const {
  return params_.at(kSideInfo).GetInt32(1);
}

int32_t GetNodesRequest::Epoch() const {
  return params_.at(kSideInfo).GetInt32(2);
}

GetNodesResponse::GetNodesResponse() : OpResponse() {
}

void GetNodesResponse::SetMembers() {
  node_ids_ = &(tensors_[kNodeIds]);
}

void GetNodesResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  GetNodesResponse& res = static_cast<GetNodesResponse&>(right);
  std::swap(node_ids_, res.node_ids_);
}

void GetNodesResponse::Init(int32_t batch_size) {
  ADD_TENSOR(tensors_, kNodeIds, kInt64, batch_size);
  node_ids_ = &(tensors_[kNodeIds]);
}

void GetNodesResponse::Append(int64_t node_id) {
  node_ids_->AddInt64(node_id);
  ++batch_size_;
}

const int64_t* GetNodesResponse::NodeIds() const {
  return node_ids_->GetInt64();
}

LookupEdgesRequest::LookupEdgesRequest()
    : OpRequest(), cursor_(0) {
}

LookupEdgesRequest::LookupEdgesRequest(const std::string& edge_type)
    : OpRequest(), cursor_(0) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("LookupEdges");

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);

  ADD_TENSOR(params_, kEdgeType, kString, 1);
  params_[kEdgeType].AddString(edge_type);

  ADD_TENSOR(tensors_, kEdgeIds, kInt64, kReservedSize);
  edge_ids_ = &(tensors_[kEdgeIds]);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
}

OpRequest* LookupEdgesRequest::Clone() const {
  LookupEdgesRequest* req = new LookupEdgesRequest(EdgeType());
  return req;
}

void LookupEdgesRequest::SetMembers() {
  edge_ids_ = &(tensors_[kEdgeIds]);
  src_ids_ = &(tensors_[kSrcIds]);
}

void LookupEdgesRequest::Set(const int64_t* edge_ids, const int64_t* src_ids,
                             int32_t batch_size) {
  edge_ids_->AddInt64(edge_ids, edge_ids + batch_size);
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
}

const std::string& LookupEdgesRequest::EdgeType() const {
  return params_.at(kEdgeType).GetString(0);
}

int32_t LookupEdgesRequest::Size() const {
  return src_ids_->Size();
}

bool LookupEdgesRequest::Next(int64_t* edge_id, int64_t* src_id) {
  if (cursor_ >= Size()) {
    return false;
  }

  *edge_id = edge_ids_->GetInt64(cursor_);
  *src_id = src_ids_->GetInt64(cursor_);
  ++cursor_;
  return true;
}

LookupNodesRequest::LookupNodesRequest()
    : OpRequest(), cursor_(0) {
}

LookupNodesRequest::LookupNodesRequest(const std::string& node_type)
    : OpRequest(), cursor_(0) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("LookupNodes");

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kNodeIds);

  ADD_TENSOR(params_, kNodeType, kString, 1);
  params_[kNodeType].AddString(node_type);

  ADD_TENSOR(tensors_, kNodeIds, kInt64, kReservedSize);
  node_ids_ = &(tensors_[kNodeIds]);
}

OpRequest* LookupNodesRequest::Clone() const {
  return new LookupNodesRequest(NodeType());
}

void LookupNodesRequest::SetMembers() {
  node_ids_ = &(tensors_[kNodeIds]);
}

void LookupNodesRequest::Set(const int64_t* node_ids, int32_t batch_size) {
  node_ids_->AddInt64(node_ids, node_ids + batch_size);
}

const std::string& LookupNodesRequest::NodeType() const {
  return params_.at(kNodeType).GetString(0);
}

int32_t LookupNodesRequest::Size() const {
  return node_ids_->Size();
}

bool LookupNodesRequest::Next(int64_t* node_id) {
  if (cursor_ >= Size()) {
    return false;
  }

  *node_id = node_ids_->GetInt64(cursor_);
  ++cursor_;
  return true;
}

LookupResponse::LookupResponse()
    : OpResponse(), info_(nullptr) {
}

LookupResponse::~LookupResponse() {
  if (is_parse_from_) {
    delete info_;
  }
}

void LookupResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  LookupResponse& res = static_cast<LookupResponse&>(right);
  std::swap(info_, res.info_);
  std::swap(infos_, res.infos_);
  std::swap(weights_, res.weights_);
  std::swap(labels_, res.labels_);
  std::swap(i_attrs_, res.i_attrs_);
  std::swap(f_attrs_, res.f_attrs_);
  std::swap(s_attrs_, res.s_attrs_);
}

void LookupResponse::SetMembers() {
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

void LookupResponse::SetSideInfo(const io::SideInfo* info, int32_t batch_size) {
  info_ = const_cast<io::SideInfo*>(info);
  batch_size_ = batch_size;

  ADD_TENSOR(params_, kSideInfo, kInt32, 4);
  infos_ = &(params_[kSideInfo]);
  infos_->AddInt32(info_->format);
  infos_->AddInt32(info_->i_num);
  infos_->AddInt32(info_->f_num);
  infos_->AddInt32(info_->s_num);

  if (info_->IsWeighted()) {
    ADD_TENSOR(tensors_, kWeightKey, kFloat, batch_size_);
    weights_ = &(tensors_[kWeightKey]);
  }
  if (info_->IsLabeled()) {
    ADD_TENSOR(tensors_, kLabelKey, kInt32, batch_size_);
    labels_ = &(tensors_[kLabelKey]);
  }
  if (info_->i_num > 0) {
    ADD_TENSOR(tensors_, kIntAttrKey, kInt64, batch_size_ * info_->i_num);
    i_attrs_ = &(tensors_[kIntAttrKey]);
  }
  if (info_->f_num > 0) {
    ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, batch_size_ * info_->f_num);
    f_attrs_ = &(tensors_[kFloatAttrKey]);
  }
  if (info_->s_num > 0) {
    ADD_TENSOR(tensors_, kStringAttrKey, kString, batch_size_ * info_->s_num);
    s_attrs_ = &(tensors_[kStringAttrKey]);
  }
}

void LookupResponse::AppendWeight(float weight) {
  if (info_->IsWeighted()) {
    weights_->AddFloat(weight);
  }
}

void LookupResponse::AppendLabel(int32_t label) {
  if (info_->IsLabeled()) {
    labels_->AddInt32(label);
  }
}

void LookupResponse::AppendAttribute(const io::AttributeValue* value) {
  if (info_->IsAttributed()) {
    for (int32_t i = 0; i < info_->i_num; ++i) {
      i_attrs_->AddInt64(value->i_attrs[i]);
    }
    for (int32_t i = 0; i < info_->f_num; ++i) {
      f_attrs_->AddFloat(value->f_attrs[i]);
    }
    for (int32_t i = 0; i < info_->s_num; ++i) {
      s_attrs_->AddString(value->s_attrs[i]);
    }
  }
}

int32_t LookupResponse::Format() const {
  return infos_->GetInt32(0);
}

int32_t LookupResponse::IntAttrNum() const {
  return infos_->GetInt32(1);
}

int32_t LookupResponse::FloatAttrNum() const {
  return infos_->GetInt32(2);
}

int32_t LookupResponse::StringAttrNum() const {
  return infos_->GetInt32(3);
}

const float* LookupResponse::Weights() const {
  return weights_->GetFloat();
}

const int32_t* LookupResponse::Labels() const {
  return labels_->GetInt32();
}

const int64_t* LookupResponse::IntAttrs() const {
  return i_attrs_->GetInt64();
}

const float* LookupResponse::FloatAttrs() const {
  return f_attrs_->GetFloat();
}

const std::string* LookupResponse::StringAttrs() const {
  return s_attrs_->GetString();
}

LookupEdgesResponse::LookupEdgesResponse()
    : LookupResponse() {
}

LookupNodesResponse::LookupNodesResponse()
    : LookupResponse() {
}

REGISTER_REQUEST(GetEdges, GetEdgesRequest, GetEdgesResponse);
REGISTER_REQUEST(GetNodes, GetNodesRequest, GetNodesResponse);
REGISTER_REQUEST(LookupEdges, LookupEdgesRequest, LookupEdgesResponse);
REGISTER_REQUEST(LookupNodes, LookupNodesRequest, LookupNodesResponse);

}  // namespace graphlearn
