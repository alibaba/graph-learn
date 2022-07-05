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

#ifndef GRAPHLEARN_CORE_PARTITION_STITCHER_H_
#define GRAPHLEARN_CORE_PARTITION_STITCHER_H_

#include <cstdint>
#include <vector>
#include "include/op_request.h"

namespace graphlearn {

template<class T>
class Stitcher {
public:
  Stitcher() {}
  virtual ~Stitcher() = default;

  virtual void Stitch(ShardsPtr<T> shards, T* t) {
    int32_t shard_id = 0;
    T* tmp = nullptr;
    if (!shards->Next(&shard_id, &tmp)) {
      return;
    }

    shards->ResetNext();

    if (shards->Size() == 1) {
      t->Swap(*tmp);
    } else if (tmp->IsSparse()) {
      t->SetSparseFlag();
      StitchSparse(shards, t);
    } else {
      StitchDense(shards, t);
    }
  }

private:
  void StitchDense(ShardsPtr<T> shards, T* t) {
    InitDense(shards, t);

    int32_t shard_id = 0;
    T* tmp = nullptr;
    // while loop each shard
    while (shards->Next(&shard_id, &tmp)) {
      auto sticker = shards->StickerPtr()->At(shard_id);
      int32_t bs = tmp->batch_size_;
      if (bs == -1) {
        bs = sticker.size();
      }

      // for loop all the data in this shard
      for (int32_t i = 0; i < bs; ++i) {
        // for loop all the tensors
        for (auto& it : tmp->tensors_) {
          if (it.first != kDegreeKey) {
            int32_t dim = it.second.Size() / bs;
            int32_t from_offset = i * dim;
            int32_t to_offset = sticker[i] * dim;
            CopyToResponse(&(it.second),
                           from_offset,
                           &(t->tensors_[it.first]),
                           to_offset,
                           dim);
          }
        }
      }
    }
  }

  void InitDense(ShardsPtr<T> shards, T* t) {
    int32_t shard_id = 0;
    T* tmp = nullptr;
    shards->Next(&shard_id, &tmp);

    int32_t batch_size = shards->StickerPtr()->Size();
    auto sticker = shards->StickerPtr()->At(shard_id);
    int32_t bs = tmp->batch_size_;
    if (bs == -1 ) {
      bs = sticker.size();
    }
    t->batch_size_ = batch_size;
    t->params_ = tmp->params_;
    t->tensors_.reserve(tmp->tensors_.size());
    for (auto& it : tmp->tensors_) {
      if (it.first != kDegreeKey) {
        int32_t dim = it.second.Size() / bs;
        ADD_TENSOR(t->tensors_, it.first, it.second.DType(), batch_size * dim);
        t->tensors_[it.first].Resize(batch_size * dim);
      }
    }

    shards->ResetNext();
  }

  void StitchSparse(ShardsPtr<T> shards, T* t) {
    std::vector<int32_t> incremental_degrees;
    int32_t batch_degree = InitSparse(shards, t, &incremental_degrees);
    Tensor& degrees_tensor = t->tensors_[kDegreeKey];
    auto degrees = degrees_tensor.GetInt32();

    int32_t shard_id = 0;
    T* tmp = nullptr;
    while (shards->Next(&shard_id, &tmp)) {
      int32_t from_offset = 0;
      auto sticker = shards->StickerPtr()->At(shard_id);
      for (int32_t i = 0; i < tmp->batch_size_; ++i) {
        int32_t offset = sticker[i];
        int32_t dim = degrees[offset];
        int32_t to_offset = incremental_degrees[offset];
        for (auto& it : tmp->tensors_) {
          if (it.first != kDegreeKey) {
            CopyToResponse(&(it.second),
                           from_offset,
                           &(t->tensors_[it.first]),
                           to_offset,
                           dim);
          }
        }
        from_offset += dim;
      }
    }
  }

  int32_t InitSparse(ShardsPtr<T> shards, T* t,
                     std::vector<int32_t>* incremental_degrees) {
    int32_t batch_size = shards->StickerPtr()->Size();

    int32_t shard_id = 0;
    T* tmp = nullptr;
    shards->Next(&shard_id, &tmp);
    t->params_ = tmp->params_;
    t->tensors_.reserve(tmp->tensors_.size());
    ADD_TENSOR(t->tensors_, kDegreeKey, kInt32, batch_size);
    t->tensors_[kDegreeKey].Resize(batch_size);
    Tensor& to_degrees_tensor = t->tensors_[kDegreeKey];
    shards->ResetNext();
    int32_t batch_degree = 0;
    while (shards->Next(&shard_id, &tmp)) {
      auto stickers = shards->StickerPtr()->At(shard_id);
      Tensor& from_degrees_tensor = tmp->tensors_[kDegreeKey];
      auto from_degrees = from_degrees_tensor.GetInt32();
      for (int32_t i = 0; i < tmp->batch_size_; ++i) {
        int32_t offset = stickers[i];
        to_degrees_tensor.SetInt32(offset, from_degrees[i]);
      }
    }

    auto to_degrees = to_degrees_tensor.GetInt32();
    incremental_degrees->resize(batch_size, 0);
    for (int32_t idx = 1; idx < batch_size; ++idx) {
      (*incremental_degrees)[idx] =
        (*incremental_degrees)[idx - 1] + to_degrees[idx - 1];
    }
    batch_degree =
      incremental_degrees->back() + to_degrees[batch_size - 1];

    t->batch_size_ = batch_size;

    for (auto& it : tmp->tensors_) {
      if (it.first != kDegreeKey) {
        ADD_TENSOR(t->tensors_, it.first, it.second.DType(), batch_degree);
        t->tensors_[it.first].Resize(batch_degree);
      }
    }

    shards->ResetNext();
    return batch_degree;
  }

  void CopyToResponse(Tensor* from, int32_t from_offset,
                      Tensor* to, int32_t to_offset,
                      int32_t dim) {
#define CASE_COPY(type, from, from_offset, to, to_offset, dim)        \
  case k##type:                                                       \
    for (int32_t i = 0; i < dim; ++i) {                               \
      to->Set##type(to_offset + i, from->Get##type(from_offset + i)); \
    }                                                                 \
    break

    DataType type = from->DType();
    switch (type) {
      CASE_COPY(Int64, from, from_offset, to, to_offset, dim);
      CASE_COPY(Int32, from, from_offset, to, to_offset, dim);
      CASE_COPY(Float, from, from_offset, to, to_offset, dim);
      CASE_COPY(Double, from, from_offset, to, to_offset, dim);
      CASE_COPY(String, from, from_offset, to, to_offset, dim);
      default:
        break;
    }
#undef CASE_COPY
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_PARTITION_STITCHER_H_
