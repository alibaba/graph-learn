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

#ifndef GRAPHLEARN_CONTRIB_KNN_KNN_REQUEST_H_
#define GRAPHLEARN_CONTRIB_KNN_KNN_REQUEST_H_

#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <vector>
#include "graphlearn/include/op_request.h"

namespace graphlearn {

class OpRequestPb;

class KnnRequest : public OpRequest {
public:
  KnnRequest();
  KnnRequest(const std::string& type, int32_t k);
  ~KnnRequest();

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;

  ShardsPtr<OpRequest> Partition() const override;

  void Set(const float* inputs, int32_t batch_size, int32_t dimension);

  const std::string& Type() const;
  int32_t K() const;
  int32_t BatchSize() const;
  int32_t Dimension() const;
  const float* Inputs() const;

private:
  std::mutex   mtx_;
  OpRequestPb* pb_;
  mutable OpRequest* clone_;
};

class KnnResponse : public OpResponse {
public:
  KnnResponse();
  ~KnnResponse() = default;

  OpResponse* New() const override {
    return new KnnResponse;
  }

  void Stitch(ShardsPtr<OpResponse> shards) override;

  void Init(int32_t batch_size, int32_t k);

  int32_t BatchSize() const;
  int32_t K() const;
  const int64_t* Ids() const;
  const float* Distances() const;

private:
  void Merge(const std::vector<KnnResponse*>& responses);
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_KNN_REQUEST_H_
