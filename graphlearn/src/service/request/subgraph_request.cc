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

namespace {
  int32_t kReservedSize = 64;
}  // anonymous namespace

SubGraphRequest::SubGraphRequest()
   : OpRequest(), batch_size_(0), src_ids_(nullptr) {
  DisableShard();
}

SubGraphRequest::SubGraphRequest(
    const std::string& nbr_type,
    const std::vector<int32_t>& num_nbrs,
    bool need_dist) : OpRequest(), batch_size_(0), src_ids_(nullptr) {
  DisableShard();
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("SubGraphSampler");

  ADD_TENSOR(params_, kNbrType, kString, 1);
  params_[kNbrType].AddString(nbr_type);

  ADD_TENSOR(params_, kNeighborCount, kInt32, num_nbrs.size());
  params_[kNeighborCount].AddInt32(num_nbrs.data(), num_nbrs.data() + num_nbrs.size());

  ADD_TENSOR(params_, kNeedDist, kInt32, 1);
  params_[kNeedDist].AddInt32(need_dist ? 1 : 0);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
}

void SubGraphRequest::Init(const Tensor::Map& params) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(params.at(kOpName).GetString(0));

  ADD_TENSOR(params_, kNbrType, kString, 1);
  params_[kNbrType].AddString(params.at(kNbrType).GetString(0));

  const auto& nbr_count_tensor = params.at(kNeighborCount);
  ADD_TENSOR(params_, kNeighborCount, kInt32, nbr_count_tensor.Size());
  params_[kNeighborCount].AddInt32(nbr_count_tensor.GetInt32(),
      nbr_count_tensor.GetInt32() + nbr_count_tensor.Size());

  ADD_TENSOR(params_, kNeedDist, kInt32, 1);
  params_[kNeedDist].AddInt32(params.at(kNeedDist).GetInt32(0));

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
}

OpRequest* SubGraphRequest::Clone() const {
  SubGraphRequest* req = new SubGraphRequest(NbrType(), GetNumNbrs(), NeedDist());
  req->DisableShard();
  return req;
}

void SubGraphRequest::Set(const int64_t* src_id,
                          int32_t batch_size) {
  src_ids_->AddInt64(src_id, src_id + batch_size);
  batch_size_ = batch_size;
}

void SubGraphRequest::Set(const int64_t* src_id,
    const int64_t* dst_id, int32_t batch_size) {
  src_ids_->AddInt64(src_id, src_id + batch_size);
  src_ids_->AddInt64(dst_id, dst_id + batch_size);
  batch_size_ = batch_size * 2;
}

void SubGraphRequest::Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors) {
  const int64_t* src_ids = tensors.at(kSrcIds).GetInt64();
  batch_size_ = tensors.at(kSrcIds).Size();
  src_ids_->AddInt64(src_ids, src_ids + batch_size_);
  if (tensors.find(kDstIds) != tensors.end()) {
    const int64_t* dst_ids = tensors.at(kDstIds).GetInt64();
    src_ids_->AddInt64(dst_ids, dst_ids + batch_size_);
    batch_size_ *= 2;
  }
}

void SubGraphRequest::Finalize() {
  src_ids_ = &(tensors_[kSrcIds]);
  batch_size_ = src_ids_->Size();
}

const std::string& SubGraphRequest::NbrType() const {
  return params_.at(kNbrType).GetString(0);
}

std::vector<int32_t> SubGraphRequest::GetNumNbrs() const {
  const int32_t* t_data = params_.at(kNeighborCount).GetInt32();
  int32_t t_size = params_.at(kNeighborCount).Size();
  return std::vector<int32_t>(t_data, t_data + t_size);
}

bool SubGraphRequest::NeedDist() const {
  return params_.at(kNeedDist).GetInt32(0) == 1;
}

const int64_t* SubGraphRequest::GetSrcIds() const {
  if (src_ids_ != nullptr) {
    return src_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const int32_t SubGraphRequest::BatchSize() const {
  return batch_size_;
}

SubGraphResponse::SubGraphResponse() : OpResponse() {}

void SubGraphResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  SubGraphResponse& res = static_cast<SubGraphResponse&>(right);
  std::swap(node_ids_, res.node_ids_);
  std::swap(row_indices_, res.row_indices_);
  std::swap(col_indices_, res.col_indices_);
  std::swap(edge_ids_, res.edge_ids_);
  std::swap(dist_to_src_, res.dist_to_src_);
  std::swap(dist_to_dst_, res.dist_to_dst_);
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

  ADD_TENSOR(tensors_, kDistToSrc, kInt32, batch_size);
  dist_to_src_ = &(tensors_[kDistToSrc]);

  ADD_TENSOR(tensors_, kDistToDst, kInt32, batch_size);
  dist_to_dst_ = &(tensors_[kDistToDst]);
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

void SubGraphResponse::SetDistToSrc(const int32_t* begin, int32_t size) {
  dist_to_src_->AddInt32(begin, begin + size);
}

void SubGraphResponse::SetDistToDst(const int32_t* begin, int32_t size) {
  dist_to_dst_->AddInt32(begin, begin + size);
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

const int32_t* SubGraphResponse::DistToSrc() const {
  return dist_to_src_->GetInt32();
}

const int32_t* SubGraphResponse::DistToDst() const {
  return dist_to_dst_->GetInt32();
}

void SubGraphResponse::Finalize() {
  node_ids_ = &(tensors_[kNodeIds]);
  row_indices_ = &(tensors_[kRowIndices]);
  col_indices_ = &(tensors_[kColIndices]);
  edge_ids_ = &(tensors_[kEdgeIds]);
  dist_to_src_ = &(tensors_[kDistToSrc]);
  dist_to_dst_ = &(tensors_[kDistToDst]);
}


REGISTER_REQUEST(SubGraphSampler, SubGraphRequest, SubGraphResponse);

}  // namespace graphlearn
