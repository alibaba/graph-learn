/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_INCLUDE_RANDOM_WALK_REQUEST_H_
#define GRAPHLEARN_INCLUDE_RANDOM_WALK_REQUEST_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "include/constants.h"
#include "include/op_request.h"

namespace graphlearn {

class RandomWalkRequest : public OpRequest {
public:
  RandomWalkRequest();
  RandomWalkRequest(const std::string& type, float p, float q, int32_t walk_len=1);
  ~RandomWalkRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;
  void Set(const int64_t* src_ids,
           int32_t batch_size);
  void Set(const int64_t* src_ids,
           const int64_t* parent_ids,
           int32_t batch_size,
           const int64_t* parent_neighbor_ids,
           const int32_t* parent_neighbor_indices,
           int32_t total_count);

  const bool IsDeepWalk() const;
  const std::string& Type() const;
  const float P() const;
  const float Q() const;
  const int32_t WalkLen() const;
  int32_t BatchSize() const;
  const int64_t* GetSrcIds() const;
  const int64_t* GetParentIds() const;
  const int64_t* GetParentNeighborIds() const;
  const int32_t* GetParentNeighborIndices() const;

protected:
  void SetMembers() override;
  // Params: kType, kPartitionKey, kSideInfo, kDistances
  Tensor* src_ids_;  // kSrcIds
  Tensor* parent_ids_;  // kNodeIds
  Tensor* parent_neighbor_ids_;  // kSrcIds
  Tensor* parent_neighbor_indices_;  // kNodeIds
  int32_t total_neighbor_count_;
};

class RandomWalkResponse : public OpResponse {
public:
  RandomWalkResponse();
  ~RandomWalkResponse() = default;

  OpResponse* New() const override {
    return new RandomWalkResponse;
  }

  void InitWalks(int32_t count); // init tensors
  void InitNeighborIds(int32_t count);
  void InitDegrees(int32_t count);

  void SetBatchSize(int32_t batch_size);

  void AppendWalks(const int64_t* walks, size_t size);
  void AppendNeighborIds(const int64_t* ids, size_t size);
  void AppendDegrees(const int32_t* degrees, size_t size);

  void Swap(OpResponse& right) override;

  const int64_t* GetWalks() const;
  const int64_t* GetNeighborIds() const;
  const int32_t* GetDegrees() const;


protected:
  void SetMembers() override;

private:
  Tensor* ids_;  // kNodeIds
  Tensor* neighbors_;
  Tensor* degrees_;
};


}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_RANDOM_WALK_REQUEST_H_
