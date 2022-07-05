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
#include <unordered_map>
#include <vector>
#include "include/op_request.h"

namespace graphlearn {

class SamplingRequest : public OpRequest {
public:
  SamplingRequest();
  SamplingRequest(const std::string& type,
                  const std::string& strategy,
                  int32_t neighbor_count,
                  int32_t filter_type = 0);
  ~SamplingRequest() = default;

  OpRequest* Clone() const override;

  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;
  void Set(const int64_t* src_ids, int32_t batch_size);
  void SetFilters(const int64_t* filter_ids, int32_t batch_size);

  const std::string& Type() const;
  const std::string& Strategy() const;
  int32_t BatchSize() const;
  int32_t NeighborCount() const { return neighbor_count_; }
  const int64_t* GetSrcIds() const;
  const int64_t* GetFilters() const;

protected:
  void SetMembers() override;
  int32_t neighbor_count_;
  int32_t filter_type_;
  Tensor* src_ids_;
  Tensor* filter_ids_;
};

class SamplingResponse : public OpResponse {
public:
  SamplingResponse();
  ~SamplingResponse() = default;

  OpResponse* New() const override {
    return new SamplingResponse;
  }

  void Swap(OpResponse& right) override;
  void SerializeTo(void* response) override;
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

protected:
  void SetMembers() override;

private:
  int32_t neighbor_count_;
  int32_t total_neighbor_count_;
  Tensor* neighbors_;
  Tensor* edges_;
  Tensor* degrees_;
};


class ConditionalSamplingRequest : public SamplingRequest {
public:
  ConditionalSamplingRequest();
  ConditionalSamplingRequest(const std::string& type,
                             const std::string& strategy,
                             int32_t neighbor_count,
                             const std::string& dst_node_type,
                             bool batch_share,
                             bool unique);
  ~ConditionalSamplingRequest() = default;

  OpRequest* Clone() const override;

  // For DagNodeRunner.
  void Init(const Tensor::Map& params) override;
  void Set(const Tensor::Map& tensors) override;

  void SetIds(const int64_t* src_ids,
              const int64_t* dst_ids,
              int32_t batch_size);
  void SetSelectedCols(const std::vector<int32_t>& int_cols,
                       const std::vector<float>& int_props,
                       const std::vector<int32_t>& float_cols,
                       const std::vector<float>& float_props,
                       const std::vector<int32_t>& str_cols,
                       const std::vector<float>& str_props);
  const std::string& Strategy() const;
  const std::string& DstNodeType() const;
  const bool BatchShare() const;
  const bool Unique() const;
  const int64_t* GetDstIds() const;

  const std::vector<int32_t> IntCols() const;
  const std::vector<float> IntProps() const;
  const std::vector<int32_t> FloatCols() const;
  const std::vector<float> FloatProps() const;
  const std::vector<int32_t> StrCols() const;
  const std::vector<float> StrProps() const;

protected:
  void SetMembers() override;

private:
  Tensor* dst_ids_;
  Tensor* int_cols_;
  Tensor* int_props_;
  Tensor* float_cols_;
  Tensor* float_props_;
  Tensor* str_cols_;
  Tensor* str_props_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SAMPLING_REQUEST_H_
