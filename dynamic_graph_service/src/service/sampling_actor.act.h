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

#ifndef DGS_SERVICE_SAMPLING_ACTOR_ACT_H_
#define DGS_SERVICE_SAMPLING_ACTOR_ACT_H_

#include "hiactor/core/actor-template.hh"
#include "hiactor/core/reference_base.hh"

#include "common/partition_router.h"
#include "core/storage/sample_builder.h"
#include "service/channel/sample_publisher.h"
#include "service/request/admin_request.h"

namespace dgs {

class SamplingActor_ref;

class ANNOTATION(actor:impl) SamplingActor
    : public hiactor::actor {
public:
  SamplingActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~SamplingActor() override = default;

  seastar::future<act::Void> ANNOTATION(actor:method)
  ApplyGraphUpdates(io::RecordBatch&& batch);

  seastar::future<act::Integer> ANNOTATION(actor:method)
  UpdateSubsRules(io::SubsRuleBatch&& batch);

  seastar::future<act::Void> ANNOTATION(actor:method)
  ExecuteAdminOperation(AdminRequest&& req);

  // Do work
  ACTOR_DO_WORK()

private:
  void InitializeImpl(const AdminRequest& req);
  void PauseExecutionImpl(const AdminRequest& req) {}
  void ResumeExecutionImpl(const AdminRequest& req) {}

  void CollectDownstreamSubsRules(
    const std::vector<storage::KVPair>& sampled_batch,
    const std::vector<storage::SubsInfo>& sub_infos);

private:
  struct RuleBufferHandle {
    RuleBufferHandle();
    ~RuleBufferHandle();

    void Push(const io::SubsRule& rule);
    seastar::future<> SendAll();

    ShardId GetGlobalShardId(PartitionId pid) {
      return partition_router_.GetGlobalShardId(pid);
    }

    SamplingActor_ref* GetSamplingActorRef(ShardId gsid) {
      return actor_refs_[gsid];
    }

    void UpdateStorePartitionInfo(uint32_t partition_num,
                                  const std::string& strategy);

    void UpdatePartitionRouter(const std::vector<RoutingUpdate>& updates) {
      partition_router_.UpdatePartitionRoutingInfo(updates);
    }

    void SetPartitionRouter(PartitionRouter&& router) {
      partition_router_ = std::move(router);
    }

    PartitionId GetStorePartitionId(VertexId vid) {
      return partitioner_.GetPartitionId(vid);
    }

  private:
    const uint32_t                   gshard_num_;
    std::vector<io::SubsRuleBatch>   buffers_;  // at partition granularity
    std::vector<SamplingActor_ref*>  actor_refs_;
    Partitioner                      partitioner_;
    PartitionRouter                  partition_router_;
  };

  using OpRelMap = std::unordered_map<OperatorId, std::vector<OperatorId>>;

private:
  storage::SampleStore*       sample_store_;
  storage::SampleBuilder*     sample_builder_;
  storage::SubscriptionTable* subs_table_;
  SamplePublisher             sample_publisher_;
  RuleBufferHandle            rule_buf_handle_;
  std::vector<bool>           is_esampler_op_map_;
  OpRelMap                    downstream_op_ids_;
  uint64_t                    num_graph_updates_;
  uint64_t                    num_rule_updates_;
  const uint64_t              data_log_period_;
  const uint64_t              rule_log_period_;
  const uint32_t              local_shard_id_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_SAMPLING_ACTOR_ACT_H_
