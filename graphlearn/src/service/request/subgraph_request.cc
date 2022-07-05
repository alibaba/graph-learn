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

#include "include/subgraph_request.h"

#include "include/constants.h"

namespace graphlearn {

SubGraphRequest::SubGraphRequest() : OpRequest() {}

SubGraphRequest::SubGraphRequest(const std::string& seed_type,
                                 const std::string& nbr_type,
                                 const std::string& strategy,
                                 int32_t batch_size,
                                 int32_t epoch) : OpRequest() {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kSeedType, kString, 1);
  params_[kSeedType].AddString(seed_type);

  ADD_TENSOR(params_, kSideInfo, kInt32, 2);
  params_[kSideInfo].AddInt32(batch_size);
  params_[kSideInfo].AddInt32(epoch);

  ADD_TENSOR(params_, kNbrType, kString, 1);
  params_[kNbrType].AddString(nbr_type);
}

const std::string& SubGraphRequest::SeedType() const {
  return params_.at(kSeedType).GetString(0);
}

const std::string& SubGraphRequest::Strategy() const {
  return params_.at(kOpName).GetString(0);
}

int32_t SubGraphRequest::BatchSize() const {
  return params_.at(kSideInfo).GetInt32(0);
}

int32_t SubGraphRequest::Epoch() const {
  return params_.at(kSideInfo).GetInt32(1);
}

const std::string& SubGraphRequest::NbrType() const {
  return params_.at(kNbrType).GetString(0);
}

SubGraphResponse::SubGraphResponse() : OpResponse() {}

void SubGraphResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  SubGraphResponse& res = static_cast<SubGraphResponse&>(right);
  std::swap(node_ids_, res.node_ids_);
  std::swap(row_indices_, res.row_indices_);
  std::swap(col_indices_, res.col_indices_);
  std::swap(edge_ids_, res.edge_ids_);
}

void SubGraphResponse::Init(int32_t batch_size) {
  ADD_TENSOR(tensors_, kNodeIds, kInt64, batch_size);
  node_ids_ = &(tensors_[kNodeIds]);

  ADD_TENSOR(tensors_, kRowIndices, kInt32, batch_size * batch_size);
  row_indices_ = &(tensors_[kRowIndices]);

  ADD_TENSOR(tensors_, kColIndices, kInt32, batch_size * batch_size);
  col_indices_ = &(tensors_[kColIndices]);
  ADD_TENSOR(tensors_, kEdgeIds, kInt64, batch_size * batch_size);
  edge_ids_ = &(tensors_[kEdgeIds]);
}

void SubGraphResponse::SetNodeIds(const int64_t* begin, int32_t size) {
  node_ids_->AddInt64(begin, begin + size);
  batch_size_ = size;
}

void SubGraphResponse::AppendEdge(int32_t row_idx,
                                  int32_t col_idx,
                                  int64_t e_id) {
  row_indices_->AddInt32(row_idx);
  col_indices_->AddInt32(col_idx);
  edge_ids_->AddInt64(e_id);
}

int32_t SubGraphResponse::EdgeCount() const {
  return row_indices_->Size();
}

const int64_t* SubGraphResponse::NodeIds() const {
  return node_ids_->GetInt64();
}

const int32_t* SubGraphResponse::RowIndices() const {
  return row_indices_->GetInt32();
}

const int32_t* SubGraphResponse::ColIndices() const {
  return col_indices_->GetInt32();
}

const int64_t* SubGraphResponse::EdgeIds() const {
  return edge_ids_->GetInt64();
}

void SubGraphResponse::SetMembers() {
  node_ids_ = &(tensors_[kNodeIds]);
  row_indices_ = &(tensors_[kRowIndices]);
  col_indices_ = &(tensors_[kColIndices]);
  edge_ids_ = &(tensors_[kEdgeIds]);
}

REGISTER_REQUEST(RandomNodeSubGraphSampler, SubGraphRequest, SubGraphResponse);
REGISTER_REQUEST(InOrderNodeSubGraphSampler, SubGraphRequest, SubGraphResponse);

}  // namespace graphlearn
