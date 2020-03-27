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

#ifndef GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_
#define GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_

#include <string>
#include "graphlearn/include/op_request.h"

namespace graphlearn {

class SamplingRequest : public OpRequest {
public:
  SamplingRequest();
  SamplingRequest(const std::string& edge_type,
                  const std::string& strategy,
                  int32_t neighbor_count);
  ~SamplingRequest() = default;

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;
  bool ParseFrom(const void* request) override;

  void Set(const int64_t* src_ids, int32_t batch_size);

  const std::string& EdgeType() const;
  const std::string& Strategy() const;
  int32_t BatchSize() const;
  int32_t NeighborCount() const { return neighbor_count_; }
  const int64_t* GetSrcIds() const;

private:
  int32_t neighbor_count_;
  Tensor* src_ids_;
};

class SamplingResponse : public OpResponse {
public:
  SamplingResponse();
  ~SamplingResponse() = default;

  OpResponse* New() const override {
    return new SamplingResponse;
  }

  void SerializeTo(void* response) override;
  bool ParseFrom(const void* response) override;
  void Stitch(ShardsPtr<OpResponse> shards) override;

  void InitNeighborIds(int32_t count);
  void InitEdgeIds(int32_t count);
  void InitDegrees(int32_t count);

  void SetBatchSize(int32_t batch_size);
  void SetNeighborCount(int32_t neighbor_count);
  void AppendNeighborId(int64_t id);
  void AppendEdgeId(int64_t id);
  void AppendDegree(int32_t degree);
  void FillWith(int64_t neighbor_id, int64_t edge_id = -1);

  int32_t BatchSize() const { return batch_size_; }
  int32_t NeighborCount() const { return neighbor_count_; }
  int32_t TotalNeighborCount() const { return total_neighbor_count_; }
  int64_t* GetNeighborIds();
  int64_t* GetEdgeIds();
  int32_t* GetDegrees();
  const int64_t* GetNeighborIds() const;
  const int64_t* GetEdgeIds() const;
  const int32_t* GetDegrees() const;

private:
  int32_t neighbor_count_;
  int32_t total_neighbor_count_;
  Tensor* neighbors_;
  Tensor* edges_;
  Tensor* degrees_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_
