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

#include "graphlearn/contrib/knn/knn_request.h"

#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/contrib/knn/config.h"
#include "graphlearn/contrib/knn/heap.h"
#include "graphlearn/include/constants.h"
#include "graphlearn/proto/service.pb.h"

namespace graphlearn {
namespace {
template<class T>
void GetTopkRet(T* heap,
                const std::vector<KnnResponse*>& responses,
                int64_t* ret_ids,
                float* ret_distances) {
  int32_t batch_size = responses[0]->BatchSize();
  int32_t k = responses[0]->K();
  for (int32_t i = 0; i < batch_size; ++i) {
    for (int32_t j = 0; j < responses.size(); ++j) {
      const int64_t* ids = responses[j]->Ids() + i * k;
      const float* distances = responses[j]->Distances() + i * k;
      for (int32_t m = 0; m < k; ++m) {
        heap->Push(distances[m], ids[m]);
      }
    }

    for (int32_t m = k; m > 0; --m) {
      heap->Pop(&(ret_distances[m - 1]), &(ret_ids[m - 1]));
    }

    ret_ids += k;
    ret_distances += k;
    heap->Clear();
  }
}
} // anonymous namespace

KnnRequest::KnnRequest() : OpRequest(), clone_(nullptr), pb_(nullptr) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("KnnOperator");
}

KnnRequest::KnnRequest(const std::string& type, int32_t k)
    : OpRequest(), clone_(nullptr), pb_(nullptr) {
  params_.reserve(5);
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("KnnOperator");

  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(type);

  ADD_TENSOR(params_, kSideInfo, kInt32, 3);
  params_[kSideInfo].AddInt32(k);
}

KnnRequest::~KnnRequest() {
  delete clone_;
  delete pb_;
}

OpRequest* KnnRequest::Clone() const {
  KnnRequest* req = new KnnRequest(Type(), K());
  req->Set(Inputs(), BatchSize(), Dimension());
  return req;
}

void KnnRequest::SerializeTo(void* request) {
  if (pb_ == nullptr) {
    ScopedLocker<std::mutex> _(&mtx_);
    if (pb_ == nullptr) {
      OpRequestPb* tmp = new OpRequestPb;
      OpRequest::SerializeTo(tmp);
      pb_ = tmp;
    }
  }

  OpRequestPb* pb = static_cast<OpRequestPb*>(request);
  pb->CopyFrom(*pb_);
}

ShardsPtr<OpRequest> KnnRequest::Partition() const {
  OpRequest* ref_this = const_cast<KnnRequest*>(this);
  ref_this->DisableShard();
  clone_ = Clone();
  clone_->DisableShard();

  ShardsPtr<OpRequest> ret(new Shards<OpRequest>(GLOBAL_FLAG(ServerCount)));
  for (int32_t i = 0; i < GLOBAL_FLAG(ServerCount); ++i) {
    if (i == GLOBAL_FLAG(ServerId)) {
      ret->Add(i, ref_this, false);
    } else {
      ret->Add(i, clone_, false);
    }
  }
  return ret;
}

void KnnRequest::Set(const float* inputs,
                     int32_t batch_size,
                     int32_t dimension) {
  params_[kSideInfo].AddInt32(batch_size);
  params_[kSideInfo].AddInt32(dimension);

  ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, batch_size * dimension);
  Tensor* t = &(tensors_[kFloatAttrKey]);
  t->AddFloat(inputs, inputs + batch_size * dimension);
}

const std::string& KnnRequest::Type() const {
  return params_.at(kType).GetString(0);
}

int32_t KnnRequest::K() const {
  return params_.at(kSideInfo).GetInt32(0);;
}

int32_t KnnRequest::BatchSize() const {
  return params_.at(kSideInfo).GetInt32(1);;
}

int32_t KnnRequest::Dimension() const {
  return params_.at(kSideInfo).GetInt32(2);;
}

const float* KnnRequest::Inputs() const {
  return tensors_.at(kFloatAttrKey).GetFloat();
}

KnnResponse::KnnResponse() : OpResponse() {
}

void KnnResponse::Init(int32_t batch_size, int32_t k) {
  ADD_TENSOR(params_, kSideInfo, kInt32, 2);
  params_[kSideInfo].AddInt32(batch_size);
  params_[kSideInfo].AddInt32(k);

  ADD_TENSOR(tensors_, kNodeIds, kInt64, batch_size * k);
  tensors_[kNodeIds].Resize(batch_size * k);

  ADD_TENSOR(tensors_, kDistances, kFloat, batch_size * k);
  tensors_[kDistances].Resize(batch_size * k);
}

int32_t KnnResponse::BatchSize() const {
  return params_.at(kSideInfo).GetInt32(0);;
}

int32_t KnnResponse::K() const {
  return params_.at(kSideInfo).GetInt32(1);;
}

void KnnResponse::Stitch(ShardsPtr<OpResponse> shards) {
  std::vector<KnnResponse*> responses;
  responses.reserve(shards->Size());

  int32_t shard_id = 0;
  OpResponse* tmp = nullptr;
  while (shards->Next(&shard_id, &tmp)) {
    responses.push_back(static_cast<KnnResponse*>(tmp));
  }

  if (responses.empty()) {
    return;
  } else if (responses.size() == 1) {
    OpResponse::Swap(*(responses[0]));
  } else {
    Merge(responses);
  }
}

void KnnResponse::Merge(const std::vector<KnnResponse*>& responses){
  int32_t batch_size = responses[0]->BatchSize();
  int32_t k = responses[0]->K();
  Init(batch_size, k);
  int64_t* ret_ids = const_cast<int64_t*>(Ids());
  float* ret_distances = const_cast<float*>(Distances());

  if (IsL2Metric()) {
    op::Heap<int64_t, op::MaxCompare>max_heap(k);
    GetTopkRet<op::Heap<int64_t, op::MaxCompare>>
      (&max_heap, responses, ret_ids, ret_distances);
  } else {
    op::Heap<int64_t, op::MinCompare>min_heap(k);
    GetTopkRet<op::Heap<int64_t, op::MinCompare>>
      (&min_heap, responses, ret_ids, ret_distances);
  }
}

const int64_t* KnnResponse::Ids() const {
  return tensors_.at(kNodeIds).GetInt64();
}

const float* KnnResponse::Distances() const {
  return tensors_.at(kDistances).GetFloat();
}

REGISTER_REQUEST(KnnOperator, KnnRequest, KnnResponse);

}  // namespace graphlearn
