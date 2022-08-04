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

#ifndef DGS_SERVICE_REQUEST_ADMIN_REQUEST_H_
#define DGS_SERVICE_REQUEST_ADMIN_REQUEST_H_

#include "common/actor_wrapper.h"
#include "common/typedefs.h"
#include "common/partition_router.h"
#include "core/storage/sample_builder.h"
#include "core/storage/sample_store.h"
#include "core/storage/subscription_table.h"
#include "generated/fbs/install_query_req_generated.h"
#include "service/request/admin_operation.h"

namespace dgs {

struct AdminRequest {
  struct Payload {
    virtual ~Payload() = default;
  };

  AdminRequest(AdminOperation op, const std::shared_ptr<Payload>& pld)
    : operation(op), payload(pld) {}

  // dump_to/load_from function should never be called.
  void dump_to(act::SerializableQueue& qu) {}  // NOLINT
  static AdminRequest load_from(act::SerializableQueue& qu) {  // NOLINT
    return AdminRequest{AdminOperation::PAUSE, {nullptr}};
  }

public:
  AdminOperation           operation;
  std::shared_ptr<Payload> payload;
};

struct SamplingInitPayload : public AdminRequest::Payload {
  explicit SamplingInitPayload(
      act::BytesBuffer&& buf,
      storage::SampleStore* sample_store,
      storage::SampleBuilder* sample_builder,
      storage::SubscriptionTable* subs_table,
      const std::string& sampling_partition_strategy,
      uint32_t sampling_partition_num,
      const std::vector<ShardId>& sampling_partition_routing_info,
      uint32_t serving_worker_num,
      const std::vector<uint32_t>& kafka_to_serving_wid)
    : AdminRequest::Payload(), buf_(std::move(buf)),
      rep_(GetMutableInstallQueryRequestRep(buf_.get_write())),
      sample_store_(sample_store),
      sample_builder_(sample_builder),
      subs_table_(subs_table),
      sampling_partition_strategy_(sampling_partition_strategy),
      sampling_partition_num_(sampling_partition_num),
      sampling_partition_routing_info_(sampling_partition_routing_info),
      serving_worker_num_(serving_worker_num),
      kafka_to_serving_wid_(kafka_to_serving_wid) {
  }

  ~SamplingInitPayload() override = default;

  QueryPriority query_priority() const {
    return rep_->priority();
  }

  const QueryPlanRep* query_plan() const {
    return rep_->query_plan();
  }

  QueryId query_id() const {
    return rep_->query_id();
  }

  storage::SampleStore* sample_store() const {
    return sample_store_;
  }

  storage::SampleBuilder* sample_builder() const {
    return sample_builder_;
  }

  storage::SubscriptionTable* subs_table() const {
    return subs_table_;
  }

  const std::string& sampling_partition_strategy() const {
    return sampling_partition_strategy_;
  }

  uint32_t sampling_partition_num() const {
    return sampling_partition_num_;
  }

  const std::vector<ShardId>& sampling_partition_routing_info() const {
    return sampling_partition_routing_info_;
  }

  uint32_t serving_worker_num() const {
    return serving_worker_num_;
  }

  const std::vector<uint32_t>& kafka_to_serving_wid_vec() const {
    return kafka_to_serving_wid_;
  }

private:
  act::BytesBuffer                buf_;
  InstallQueryRequestRep*         rep_;
  storage::SampleStore*           sample_store_;
  storage::SampleBuilder*         sample_builder_;
  storage::SubscriptionTable*     subs_table_;
  const std::string               sampling_partition_strategy_;
  const uint32_t                  sampling_partition_num_;
  const std::vector<ShardId>      sampling_partition_routing_info_;
  const uint32_t                  serving_worker_num_;
  const std::vector<uint32_t>     kafka_to_serving_wid_;
};

struct ServingInitPayload : public AdminRequest::Payload {
  explicit ServingInitPayload(
      act::BytesBuffer&& buf,
      storage::SampleStore* sample_store)
    : AdminRequest::Payload(), buf_(std::move(buf)),
      rep_(GetMutableInstallQueryRequestRep(buf_.get_write())),
      sample_store_(sample_store) {
  }

  ~ServingInitPayload() override = default;

  QueryPriority query_priority() const {
    return rep_->priority();
  }

  const QueryPlanRep* query_plan() const {
    return rep_->query_plan();
  }

  QueryId query_id() const {
    return rep_->query_id();
  }

  storage::SampleStore* sample_store() const {
    return sample_store_;
  }

private:
  act::BytesBuffer        buf_;
  InstallQueryRequestRep* rep_;
  storage::SampleStore*   sample_store_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_REQUEST_ADMIN_REQUEST_H_
