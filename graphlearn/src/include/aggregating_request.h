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

#ifndef GRAPHLEARN_INCLUDE_AGGREGATING_REQUEST_H_
#define GRAPHLEARN_INCLUDE_AGGREGATING_REQUEST_H_

#include <string>
#include "include/op_request.h"

namespace graphlearn {

class AggregatingRequest : public OpRequest {
public:
  AggregatingRequest();
  AggregatingRequest(const std::string& type,
                   const std::string& strategy);
  virtual ~AggregatingRequest() = default;

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;

  void Set(const int64_t* node_ids,
           const int32_t* segment_ids,
           int32_t num_ids,
           int32_t num_segments);

  const std::string& Type() const;
  const std::string& Strategy() const;
  bool Next(int64_t* node_id, int32_t* segment_id);
  int32_t NumIds() const { return node_ids_->Size(); }
  bool SegmentEnd(int32_t segment_id) const;
  int32_t NumSegments() const { return num_segments_; }

protected:
  void Finalize() override;

private:
  int32_t cursor_;
  Tensor* node_ids_;
  Tensor* segment_ids_;
  int32_t num_segments_;
};

class AggregatingResponse : public OpResponse {
public:
  AggregatingResponse();
  virtual ~AggregatingResponse() = default;

  OpResponse* New() const override {
    return new AggregatingResponse;
  }

  void Swap(OpResponse& right) override;

  void SetName(const std::string& name);
  void SetEmbeddingDim(int32_t dim);
  void SetNumSegments(int32_t dim);

  std::string Name() const { return name_; }

  int32_t EmbeddingDim() const { return emb_dim_; }
  void AppendEmbedding(const float* value);
  const float* Embeddings() const;

  int32_t NumSegments() const { return batch_size_; }
  void AppendSegment(int32_t size);
  const int32_t* Segments() const;

  void Stitch(ShardsPtr<OpResponse> shards) override;

protected:
  void Finalize() override;

private:
  std::string name_;
  int32_t emb_dim_;
  Tensor* embs_;
  Tensor* segments_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_AGGREGATING_REQUEST_H_
