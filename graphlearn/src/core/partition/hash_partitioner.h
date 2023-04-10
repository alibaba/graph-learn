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

#ifndef GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_
#define GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_

#include <cstdint>
#include <vector>
#include "core/partition/partitioner.h"
#include "include/config.h"
#include "include/op_request.h"

namespace graphlearn {

template<class T>
class HashPartitioner : public BasePartitioner<T> {
public:
  explicit HashPartitioner(int32_t range) : range_(range) {}
  ~HashPartitioner() = default;

  ShardsPtr<T> Partition(const T* req) override {
    ShardsPtr<T> ret(new Shards<T>(range_));

    if (!req->IsShardable()) {
      ret->Add(req->ShardId(), const_cast<T*>(req), false);
      return ret;
    }

    auto iter = req->tensors_.find(req->ShardKey());
    if (iter == req->tensors_.end()) {
      ret->Add(req->ShardId(), const_cast<T*>(req), false);
      return ret;
    }

    auto& part_key = iter->second;
    int32_t length = part_key.Size();
    auto part_by = part_key.GetInt64();

    std::unordered_map<std::string, int32_t> sparse_value_offset;
    for (const auto& it : req->sparse_tensors_) {
      sparse_value_offset.emplace(it.first, 0);
    }

    for (int64_t index = 0; index < length; ++index) {
      int32_t part_id = HashToPartId(part_by[index]);

      T* part_req = ret->AddSticker(part_id, index);
      if (part_req == nullptr) {
        part_req = InitPartRequest(req);
        ret->Add(part_id, part_req, true);
      }

      for (const auto& it : req->tensors_) {
        const Tensor* from = &(it.second);
        Tensor* target = &(part_req->tensors_[it.first]);
        size_t dim = from->Size() / length;
        CopyToPartRequest(from, index * dim,  (index  + 1) * dim, target);
      }

      for (const auto& it : req->sparse_tensors_) {
        const SparseTensor* from = &(it.second);
        SparseTensor* target = &(part_req->sparse_tensors_[it.first]);
        auto& segments = from->Segments();
        auto& values = from->Values();

        int32_t start = sparse_value_offset[it.first];
        int32_t end = start + segments.GetInt32(index);
        sparse_value_offset[it.first] = end;

        CopyToPartRequest(&segments, index, index + 1, target->MutableSegments());
        CopyToPartRequest(&values, start, end, target->MutableValues());
      }
    }
    return ret;
  }

private:
  int32_t HashToPartId(int64_t id) {
    return llabs(id) % range_;
  }

  T* InitPartRequest(const T* req) {
    T* part_req = req->Clone();
    part_req->DisableShard();

    part_req->tensors_.reserve(req->tensors_.size());
    for (const auto& it : req->tensors_) {
      auto& t = it.second;
      ADD_TENSOR(part_req->tensors_, it.first, t.DType() , t.Size());
    }
    part_req->sparse_tensors_.reserve(req->sparse_tensors_.size());
    for (const auto& it : req->sparse_tensors_) {
      auto& t = it.second;
      Tensor ind(t.Segments().DType(), it.second.Segments().Size());
      Tensor val(t.Values().DType(), it.second.Values().Size());
      part_req->sparse_tensors_.emplace(it.first, std::move(SparseTensor{std::move(ind), std::move(val)}));
    }
    return part_req;
  }

  void CopyToPartRequest(const Tensor* from,
                         int32_t start,
                         int32_t end,
                         Tensor* target) {
    DataType type = from->DType();

#define CASE_COPY(Type)                           \
  case k##Type:                                   \
    for (int32_t i = start; i < end; ++i) { \
      target->Add##Type(from->Get##Type(i));      \
    }                                             \
    break

    switch (type) {
      CASE_COPY(Int64);
      CASE_COPY(Int32);
      CASE_COPY(Float);
      CASE_COPY(Double);
      CASE_COPY(String);
      default:
        break;
    }
#undef CASE_COPY
  }

private:
  int32_t range_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_PARTITION_HASH_PARTITIONER_H_
