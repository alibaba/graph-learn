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

#include "service/sampling_actor.act.h"

#include "seastar/core/when_all.hh"

#include "common/log.h"
#include "common/options.h"
#include "core/execution/dag.h"
#include "core/execution/dag_node.h"
#include "service/actor_ref_builder.h"

namespace dgs {

SamplingActor::SamplingActor(hiactor::actor_base* exec_ctx,
                             const hiactor::byte_t* addr)
  : hiactor::stateless_actor(exec_ctx, addr),
    subs_table_(nullptr),
    sample_store_(nullptr),
    sample_builder_(nullptr),
    sample_publisher_(),
    num_graph_updates_(0),
    num_rule_updates_(0),
    data_log_period_(
      Options::GetInstance().GetLoggingOptions().data_log_period),
    rule_log_period_(
      Options::GetInstance().GetLoggingOptions().rule_log_period),
    local_shard_id_(actor::LocalShardId()) {
  auto& opts = Options::GetInstance().GetSamplePublishingOptions();
  sample_publisher_ = SamplePublisher(
      opts.kafka_topic, opts.kafka_partition_num);
  LOG(INFO) << "data logging period is " << data_log_period_;
  LOG(INFO) << "rule logging period is " << rule_log_period_;
}

seastar::future<actor::Void>
SamplingActor::ExecuteAdminOperation(AdminRequest&& req) {
  switch (req.operation) {
    case AdminOperation::PAUSE: {
      // TODO(xiaoming.qxm):
      break;
    }
    case AdminOperation::RESUME: {
      // TODO(xiaoming.qxm):
      break;
    }
    case AdminOperation::INIT: {
      InitializeImpl(req);
      break;
    }
  }
  return seastar::make_ready_future<actor::Void>();
}

void SamplingActor::InitializeImpl(const AdminRequest& req) {
  LOG(INFO) << "Initialize on global shard " << actor::GlobalShardId();
  auto *param = dynamic_cast<SamplingInitPayload*>(req.payload.get());
  // Arm sample store.
  sample_store_ = param->sample_store();
  sample_builder_ = param->sample_builder();
  subs_table_ = param->subs_table();
  // Install query.
  auto *dag = new execution::Dag(param->query_plan());
  sample_builder_->Init(dag);
  subs_table_->Init(dag);

  assert(!dag->nodes().empty());
  auto *first_node = dag->nodes()[0];
  if (first_node->id() != 0 || first_node->kind() != PlanNode::Kind_SOURCE) {
    LOG(FATAL) << "First node's id must be 0 and its kind must be SOURCE";
  }

  static_assert(std::is_same<OperatorId, int32_t>::value,
      "Operator id type should be int32");

  is_esampler_op_map_ = std::vector<bool>(dag->nodes().size(), false);
  std::unordered_map<int, OperatorId> virtual_vop_map;
  for (auto* node : dag->nodes()) {
    std::vector<OperatorId> downstreams;
    for (auto edge : node->out_edges()) {
      auto dst_id = edge->dst()->id();
      if (edge->dst()->kind() == PlanNode::Kind_VERTEX_SAMPLER) {
        if (!virtual_vop_map.count(dst_id)) {
          virtual_vop_map.emplace(dst_id, 0);
        }
        auto vprefix = virtual_vop_map[dst_id];
        // Note that we assume `node id` in dag is less than INT16_MAX.
        assert(dst_id >= 0 && dst_id <= INT16_MAX);
        downstreams.push_back(dst_id + (vprefix << 16));
        virtual_vop_map[dst_id] += 1;
      } else {
        downstreams.push_back(edge->dst()->id());
      }
    }
    if (!downstreams.empty()) {
      downstream_op_ids_.emplace(node->id(), std::move(downstreams));
    }

    assert(node->id() < is_esampler_op_map_.size());

    if (node->kind() == PlanNode::Kind_EDGE_SAMPLER) {
      is_esampler_op_map_[node->id()] = true;
    }
  }
  delete dag;

  rule_buf_handle_.UpdateStorePartitionInfo(
      param->sampling_partition_num(),
      param->sampling_partition_strategy());

  rule_buf_handle_.SetPartitionRouter(
    PartitionRouter(param->sampling_partition_routing_info()));

  sample_publisher_.UpdateSinkKafkaPartitions(
      param->pub_kafka_pids(),
      param->serving_partition_strategy(),
      param->serving_partition_num(),
      param->serving_worker_num());
}

seastar::future<actor::Void>
SamplingActor::ApplyGraphUpdates(io::RecordBatch&& input_batch) {
  if (++num_graph_updates_ % data_log_period_ == 0) {
    LOG(INFO) << "Apply Graph Update on global shard "
              << actor::GlobalShardId()
              << ": #processed_updates is "
              << num_graph_updates_
              << ", record_batch size is "
              << input_batch.GetView().RecordNum();
  }

  // 1. Sampling records using SamplerBuilder.
  auto sampled_batch = sample_builder_->Sample(input_batch);
  if (sampled_batch.empty()) {
    return seastar::make_ready_future<actor::Void>();
  }

  std::vector<storage::SubsInfo> subs_infos;
  for (uint32_t idx = 0; idx < sampled_batch.size(); ++idx) {
    auto &key = sampled_batch[idx].key;
    auto &record = sampled_batch[idx].value;
    // 2. Put sampled record into SampleStore.
    if (record.GetView().Type() == RecordType::VERTEX) {
      sample_store_->PutVertex(key, {record.Data(), record.Size()});
    } else {
      sample_store_->PutEdge(key, {record.Data(), record.Size()});
    }

    // 3. Collect subscribed partition ids.
    subs_table_->GetSubscribedWorkers(idx, key.pkey, &subs_infos);
  }

  if (subs_infos.empty()) {
    return seastar::make_ready_future<actor::Void>();
  }

  // 4. Collect subscription rules using query dependency info.
  CollectDownstreamSubsRules(sampled_batch, subs_infos);

  // 5. Send all sampled records and downstream subscription rules.
  return seastar::when_all(
      sample_publisher_.Publish(sampled_batch, subs_infos),
      rule_buf_handle_.SendAll()).then([] (auto) {
    return seastar::make_ready_future<actor::Void>();
  });
}

seastar::future<actor::Integer>
SamplingActor::UpdateSubsRules(io::SubsRuleBatch &&rule_batch) {
  // Forward the rule_batch to the shard holding the corresponding partition
  assert(!rule_batch.rules.empty());
  auto pid = rule_buf_handle_.GetStorePartitionId(rule_batch.rules[0].pkey.vid);
  auto dst_gsid = rule_buf_handle_.GetGlobalShardId(pid);
  if (actor::GlobalShardId() != dst_gsid) {
    return rule_buf_handle_.GetSamplingActorRef(dst_gsid)
        ->UpdateSubsRules(std::move(rule_batch));
  }

  if (++num_rule_updates_ % rule_log_period_ == 0) {
    LOG(INFO) << "Update Subscription Rule on global shard "
              << actor::GlobalShardId()
              << ": #processed_updates is "
              << num_rule_updates_
              << ", rule_batch size is "
              << rule_batch.rules.size();
  }

  std::vector<uint32_t> new_rule_ids;
  subs_table_->UpdateRules(rule_batch.rules, &new_rule_ids);
  if (new_rule_ids.empty()) {
    return seastar::make_ready_future<actor::Integer>(
            actor::Integer(static_cast<int32_t>(actor::GlobalShardId())));
  }

  std::vector<storage::KVPair> sampled_batch;
  std::vector<storage::KVPair> output;
  std::vector<storage::SubsInfo> subs_info_buf;
  for (auto id : new_rule_ids) {
    auto &rule = rule_batch.rules[id];
    // erase the 16 most-significant bits
    rule.pkey.op_id = static_cast<int16_t>(rule.pkey.op_id);
    if (is_esampler_op_map_[rule.pkey.op_id]) {
      sample_store_->GetEdgesByPrefix(rule.pkey, &output);
    } else {
      sample_store_->GetVerticesByPrefix(rule.pkey, &output);
    }
    for (auto& o : output) {
      sampled_batch.push_back(std::move(o));
      subs_info_buf.emplace_back(
          static_cast<uint32_t>(sampled_batch.size() - 1),
          static_cast<PartitionId>(rule.worker_id));
    }
    output.clear();
  }

  CollectDownstreamSubsRules(sampled_batch, subs_info_buf);

  return seastar::when_all(
      sample_publisher_.Publish(sampled_batch, subs_info_buf),
      rule_buf_handle_.SendAll()).then([] (auto) {
    return seastar::make_ready_future<actor::Integer>(
            actor::Integer(static_cast<int>(actor::GlobalShardId())));
  });
}

void SamplingActor::CollectDownstreamSubsRules(
    const std::vector<storage::KVPair>& sampled_batch,
    const std::vector<storage::SubsInfo>& sub_infos) {
  for (auto &info : sub_infos) {
    auto &key = sampled_batch[info.record_id].key;
    auto &record = sampled_batch[info.record_id].value;
    // FIXME(@xmqin): double-check here.
    if (record.GetView().Type() == RecordType::EDGE
          && downstream_op_ids_.count(key.pkey.op_id)) {
      auto &ds_op_ids = downstream_op_ids_[key.pkey.op_id];
      auto erecord_view = record.GetView().AsEdgeRecord();
      io::SubsRule subs_rule{erecord_view.DstType(), erecord_view.DstId(),
                             0, info.worker_id};
      for (auto ds_op_id : ds_op_ids) {
        subs_rule.pkey.op_id = ds_op_id;
        rule_buf_handle_.Push(subs_rule);
      }
    }
  }
}

SamplingActor::RuleBufferHandle::RuleBufferHandle()
  : gshard_num_(actor::GlobalShardCount()),
    partitioner_(std::move(Partitioner())) {
  // create actor references.
  actor_refs_.reserve(gshard_num_);

  for (int i = 0; i < gshard_num_; ++i) {
    auto builder = hiactor::scope_builder(i);
    actor_refs_.push_back(MakeSamplingActorInstRefPtr(builder));
  }
}

SamplingActor::RuleBufferHandle::~RuleBufferHandle() {
  for (auto ref : actor_refs_) {
    delete ref;
  }
}

void SamplingActor::RuleBufferHandle::Push(const io::SubsRule& rule) {
  auto dst_pid = partitioner_.GetPartitionId(rule.pkey.vid);
  buffers_[dst_pid].rules.push_back(rule);
}

void SamplingActor::RuleBufferHandle::UpdateStorePartitionInfo(
    uint32_t partition_num,
    const std::string& strategy) {
  try {
    auto partitioner = PartitionerFactory::Create(strategy, partition_num);
    partitioner_ = std::move(partitioner);
    LOG(INFO) << "Update partition info with partitioning strategy: "
               << strategy << ", partition num: " << partition_num;
  } catch (std::exception& ex) {
    LOG(ERROR) << "Update partition info failed: " << ex.what();
  }
  buffers_.reserve(partition_num);
  for (int i = 0; i < partition_num; ++i) {
    buffers_.emplace_back();
  }
}

seastar::future<> SamplingActor::RuleBufferHandle::SendAll() {
  std::vector<seastar::future<>> futs;
  for (PartitionId pid = 0; pid < buffers_.size(); ++pid) {
    if (!buffers_[pid].rules.empty()) {
      auto sid = GetGlobalShardId(pid);
      auto fut = actor_refs_[sid]->UpdateSubsRules(
          std::move(buffers_[pid])).then(
              [this, pid] (actor::Integer&& ret) {
        ShardId dst_sid = *reinterpret_cast<ShardId*>(&ret.val);
        // Update the remote destination ShardId for the partition
        if (dst_sid != GetGlobalShardId(pid)) {
          partition_router_.UpdatePartitionRoutingInfo(
            RoutingUpdate(pid, dst_sid));
        }
        return seastar::make_ready_future<>();
      });
      futs.emplace_back(std::move(fut));
      // populate with a new buffer.
      buffers_[pid] = io::SubsRuleBatch{};
    }
  }
  return seastar::when_all(futs.begin(), futs.end()).discard_result();
}

}  // namespace dgs
