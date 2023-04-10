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

#include "include/dag_request.h"

#include "core/dag/tape.h"
#include "include/config.h"
#include "generated/proto/request.pb.h"

namespace graphlearn {

DagRequest::DagRequest() : BaseRequest(false) {}

void DagRequest::SerializeTo(void* request) {
  DagDef* pb = static_cast<DagDef*>(request);
  def_.Swap(pb);
}

bool DagRequest::ParseFrom(const void* request) {
  DagDef* pb = const_cast<DagDef*>(static_cast<const DagDef*>(request));
  def_.Swap(pb);
  return true;
}

bool DagRequest::ParseFrom(const void* request, const bool copy) {
  DagDef* pb = const_cast<DagDef*>(static_cast<const DagDef*>(request));
  if (copy) {
    def_ = *pb;
  } else {
    def_.Swap(pb);
  }
  return true;
}

std::string DagRequest::Name() const {
  return "DagRequest";
}

GetDagValuesRequest::GetDagValuesRequest()
    : BaseRequest(false),
      id_(-1),
      client_id_(GLOBAL_FLAG(ClientId)) {
}

GetDagValuesRequest::GetDagValuesRequest(int32_t dag_id,
                                         int32_t client_id)
    : BaseRequest(false),
      id_(dag_id),
      client_id_(client_id) {
}

void GetDagValuesRequest::SerializeTo(void* request) {
  DagValuesRequestPb* pb = static_cast<DagValuesRequestPb*>(request);
  pb->set_id(id_);
  pb->set_client_id(client_id_);
}

bool GetDagValuesRequest::ParseFrom(const void* request) {
  const DagValuesRequestPb* req =
      static_cast<const DagValuesRequestPb*>(request);
  id_ = req->id();
  client_id_ = req->client_id();
  return true;
}

GetDagValuesResponse::GetDagValuesResponse()
    : epoch_(-1), index_(-1) {
}

GetDagValuesResponse::GetDagValuesResponse(GetDagValuesResponse&& res)
    : records_(std::move(res.records_)),
      epoch_(res.epoch_),
      index_(res.index_) {
}

void GetDagValuesResponse::MoveFrom(Tape* tape) {
  for (int32_t i = 1; i < tape->Size(); ++i) {
    auto& ret = tape->Retrieval(i);
    if (ret.Size() > 0) {
      records_.emplace(i, std::move(ret));
    }
  }
}

std::pair<const Tensor*, const Tensor*> GetDagValuesResponse::GetValue(
    int32_t node_id, const std::string& key) {
  auto it = records_.find(node_id);
  if (it == records_.end()) {
    return {nullptr, nullptr};
  }
  auto& tensors = it->second;
  return tensors.Find(key);;
}

void GetDagValuesResponse::SetEpoch(int32_t epoch) {
  epoch_ = epoch;
}

void GetDagValuesResponse::SetIndex(int32_t index) {
  index_ = index;
}

void GetDagValuesResponse::SerializeTo(void* response) {
  DagValuesResponsePb* pb = static_cast<DagValuesResponsePb*>(response);
  for (auto& record : records_) {
    DagNodeValue* res = pb->add_dag_node_value();
    res->set_id(record.first);
    auto& tensors = record.second.tensors_;
    auto& sparse_tensors = record.second.sparse_tensors_;
    for (auto& tensor : tensors) {
      auto& t = tensor.second;
      TensorValue* v = res->add_tensors();
      v->set_name(tensor.first);
      v->set_length(t.Size());
      v->set_dtype(static_cast<int32_t>(t.DType()));
      t.SwapWithProto(v);
    }
    for (auto& tensor : sparse_tensors) {
      auto& t = tensor.second;
      auto t_segments = t.MutableSegments();
      auto t_values = t.MutableValues();
      SparseTensorValue* v = res->add_sparse_tensors();
      v->set_name(tensor.first);

      TensorValue* segments = v->mutable_segments();
      segments->set_name("segments");
      segments->set_length(t_segments->Size());
      segments->set_dtype(static_cast<int32_t>(t_segments->DType()));
      t_segments->SwapWithProto(segments);

      TensorValue* values = v->mutable_values();
      values->set_name("values");
      values->set_length(t_values->Size());
      values->set_dtype(static_cast<int32_t>(t_values->DType()));
      t_values->SwapWithProto(values);
    }
  }
  pb->set_epoch(epoch_);
  pb->set_index(index_);
}

bool GetDagValuesResponse::ParseFrom(const void* response) {
  DagValuesResponsePb* pb = const_cast<DagValuesResponsePb*>(
    static_cast<const DagValuesResponsePb*>(response));
  for (int32_t i = 0; i < pb->dag_node_value_size(); ++i) {
    DagNodeValue* res = pb->mutable_dag_node_value(i);
    Tensor::Map mmap;
    SparseTensor::Map smap;
    for (int32_t j = 0; j < res->tensors_size(); ++j) {
      TensorValue* v = res->mutable_tensors(j);
      Tensor t(static_cast<DataType>(v->dtype()));
      t.SwapWithProto(v);
      mmap.emplace(v->name(), std::move(t));
    }
    for (int32_t j = 0; j < res->sparse_tensors_size(); ++ j) {
      SparseTensorValue* sv = res->mutable_sparse_tensors(j);
      auto sv_values = sv->mutable_values();
      auto sv_segments = sv->mutable_segments();
      Tensor values(static_cast<DataType>(sv_values->dtype()));
      values.SwapWithProto(sv_values);

      Tensor segments(static_cast<DataType>(sv_segments->dtype()));
      segments.SwapWithProto(sv_segments);
      smap.emplace(sv->name(), SparseTensor{std::move(segments), std::move(values)});
    }
    records_.emplace(res->id(), TensorMap{std::move(mmap), std::move(smap)});
  }
  epoch_ = pb->epoch();
  index_ = pb->index();
  return true;
}

}  // namespace graphlearn
