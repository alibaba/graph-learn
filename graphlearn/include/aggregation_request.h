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

#ifndef GRAPHLEARN_INCLUDE_AGGREGATION_REQUEST_H_
#define GRAPHLEARN_INCLUDE_AGGREGATION_REQUEST_H_

#include <string>
#include "graphlearn/include/graph_request.h"

namespace graphlearn {

class AggregateNodesRequest : public OpRequest {
public:
  AggregateNodesRequest();
  AggregateNodesRequest(const std::string& node_type,
                        const std::string& strategy);
  virtual ~AggregateNodesRequest() = default;

  OpRequest* Clone() const override;
  void SerializeTo(void* request) override;
  bool ParseFrom(const void* request) override;

  void Set(const int64_t* node_ids,
           int32_t num_ids,
           const int32_t* segments,
           int32_t num_segments);

  const std::string& NodeType() const;
  bool NextId(int64_t* node_id);
  int32_t NumIds() const { return num_ids_; }
  bool NextSegment(int32_t* segment);
  int32_t NumSegments() const { return num_segments_; }

private:
  int32_t id_cursor_;
  int32_t num_ids_;
  int32_t segment_cursor_;
  int32_t num_segments_;
  Tensor* node_ids_;
  Tensor* segments_;
};

class AggregateNodesResponse : public LookupResponse {
public:
  AggregateNodesResponse();
  AggregateNodesResponse(const io::SideInfo* info, int32_t batch_size);
  virtual ~AggregateNodesResponse() = default;

  OpResponse* New() const override {
    return new AggregateNodesResponse;
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_AGGREGATION_REQUEST_H_
