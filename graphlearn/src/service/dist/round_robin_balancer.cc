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

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "common/base/errors.h"
#include "common/base/log.h"
#include "service/dist/load_balancer.h"

namespace graphlearn {

class RoundRobinBalancer : public LoadBalancer {
public:
  explicit RoundRobinBalancer(int32_t resource_num)
      : LoadBalancer(resource_num),
        part_num_(0),
        replica_(1) {
  }

  ~RoundRobinBalancer() = default;

  Status Calc(int32_t part_num, int32_t replica) override {
    if (part_num <= 0 || replica <= 0) {
      LOG(WARNING) << "Invalid balancer parameter, part:"
                   << part_num << " replica:" << replica;
      return error::InvalidArgument("Invalid balancer parameter");
    }

    if (resource_num_ <= 0) {
      LOG(WARNING) << "Invalid balancer resource:" << resource_num_;
      return error::Unavailable("No resource available");
    }

    if (part_num_ != part_num || replica_ != replica) {
      part_num_ = part_num;
      replica_ = (replica < resource_num_ ? replica : resource_num_);
      parts_.clear();

      if (resource_num_ >= part_num_) {
        DownDistribute();
      } else {
        UpDistribute();
      }
    }

    return Status::OK();
  }

  Status GetPart(int32_t part_id, std::vector<int32_t>* resources) override {
    assert(resources != nullptr);
    if (part_num_ == 0) {
      return error::Unavailable("Please call Calc() first.");
    }
    if (part_id >= part_num_) {
      LOG(WARNING) << "Invalid part_id: " << part_id
                   << ", part_num: " << part_num_;
      return error::InvalidArgument("Invalid part id");
    }

    auto it = parts_.find(part_id);
    if (it == parts_.end()) {
      return error::Unavailable("Please call Calc() first.");
    }
    *resources = it->second;
    return Status::OK();
  }

private:
  void DownDistribute() {
    assert(part_num_ > 0);
    assert(resource_num_ > 0);
    // round 1: assign [0, count*part_num) resources
    int32_t count = resource_num_ / part_num_;
    for (int32_t i = 0; i < part_num_; ++i) {
      auto& part = parts_[i];
      for (int32_t j = 0; j < count; ++j) {
        part.push_back(j + i * count);
      }
    }
    int32_t assigned = part_num_ * count;
    int32_t remainder = resource_num_ - assigned;
    // round 2: assign [count*part_num, resource_num) resources
    if (remainder > 0) {
      for (int i = 0, j = assigned; j < resource_num_; ++i, ++j) {
        parts_[i].push_back(j);
      }
    }
    for (int32_t i = 0; i < part_num_; ++i) {
      auto& current = parts_[i];
      if (current.size() < replica_) {
        // round 3: merge resources from right to current
        int32_t count = replica_ - current.size();
        int32_t right = (i < part_num_ - 1 ? i + 1 : 0);
        for (int32_t j = 0; j < count;) {
          auto& next = parts_[right];
          for (int32_t k = 0; k < next.size(); ++k) {
            current.push_back(next[k]);
            if (++j >= count) {
              break;
            }
          }
          right = (right < part_num_ - 1 ? right + 1 : 0);
        }
      }
    }
  }

  void UpDistribute() {
    assert(part_num_ > 0);
    assert(resource_num_ > 0);
    // round 1: assign [0, resource_num) resources.
    int32_t count = part_num_ / resource_num_;
    for (int32_t i = 0, j = 0, sum = 0; i < part_num_; ++i) {
      auto& part = parts_[i];
      part.push_back(j);
      if (++sum >= count) {
        sum = 0;
        j = (j < resource_num_ - 1 ? j + 1 : 0);
      }
    }
    for (int32_t i = 0; i < part_num_; ++i) {
      auto& current = parts_[i];
      if (current.size() < replica_) {
        // round 2: merge resources from next(current + 1) to current
        int32_t next_resource = (current[0] < resource_num_ - 1 ?
            current[0] + 1 : 0);
        for (int32_t count = replica_ - current.size();
             count > 0;
             --count) {
          current.push_back(next_resource);
          next_resource = (next_resource < resource_num_ - 1 ?
              next_resource + 1 : 0);
        }
      }
    }
  }

private:
  int32_t part_num_;
  int32_t replica_;
  std::unordered_map<int32_t, std::vector<int32_t>> parts_;
};

LoadBalancer* NewRoundRobinBalancer(int32_t resource_num) {
  return new RoundRobinBalancer(resource_num);
}

}  // namespace graphlearn
