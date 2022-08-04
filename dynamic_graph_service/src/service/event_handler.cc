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

#include "service/event_handler.h"

#include "hiactor/core/reference_base.hh"
#include "seastar/http/handlers.hh"

#include "common/options.h"
#include "common/utils.h"
#include "service/actor_ref_builder.h"
#include "service/adaptive_rate_limiter.h"
#include "service/serving_group.actg.h"

namespace dgs {

using HttpReplyPtr = std::unique_ptr<seastar::httpd::reply>;

class RunQueryHandler : public seastar::httpd::handler_base {
public:
  RunQueryHandler(hiactor::scope_builder& builder,  // NOLINT
                  AdaptiveRateLimiter* rate_limiter)
    : local_shard_id_(act::LocalShardId()), num_processed_requests_(0),
      logging_period_(
        Options::GetInstance().GetLoggingOptions().request_log_period),
      available_request_units_(
        Options::GetInstance().GetEventHandlerOptions().max_local_requests),
      nls_cursor_(0), non_local_shard_ids_(), rate_limiter_(rate_limiter) {
    non_local_shard_ids_.reserve(act::LocalShardCount() - 1);
    actor_refs_.reserve(act::LocalShardCount());
    for (unsigned i = 0; i < act::LocalShardCount(); i++) {
      auto g_sid = act::GlobalShardIdAnchor() + i;
      builder.set_shard(g_sid);
      actor_refs_.emplace_back(MakeServingActorInstRef(builder));
      if (i != local_shard_id_) {
        non_local_shard_ids_.emplace_back(i);
      }
    }

    LOG(INFO) << "RunQueryHandler is constructed on local shard "
              << local_shard_id_ << ", initial available request units is "
              << available_request_units_ << ", logging period is "
              << logging_period_;
  }

  ~RunQueryHandler() override = default;

  seastar::future<std::unique_ptr<seastar::httpd::reply>>
  handle(const seastar::sstring& path,
         std::unique_ptr<seastar::httpd::request> req,
         std::unique_ptr<seastar::httpd::reply> rep) override {
    auto qid_it = req->query_parameters.find("qid");
    auto vid_it = req->query_parameters.find("vid");
    assert(qid_it != req->query_parameters.end());
    assert(vid_it != req->query_parameters.end());

    QueryId qid = std::stoi(qid_it->second);
    VertexId vid = std::stoi(vid_it->second);
    start_time_ = CurrentTimeInMs();
    auto dst_shard_id = GetDestShardId();
    return actor_refs_[dst_shard_id].RunQuery(RunQueryRequest(qid, vid)).then(
        [rep = std::move(rep), this, dst_shard_id] (QueryResponse&& res) mutable {  // NOLINT
      if (dst_shard_id == local_shard_id_) {
        ++available_request_units_;
      }
      uint64_t elapsed_time = CurrentTimeInMs() - start_time_;
      rate_limiter_->SubmitLatency(local_shard_id_, elapsed_time);

      if (++num_processed_requests_ % logging_period_ == 0) {
        LOG(INFO) << "#processed requests is " << num_processed_requests_
                  << ", elapsed time for current query is "
                  << elapsed_time << " ms, #available_request_units is "
                  << available_request_units_ << " on local shard "
                  << local_shard_id_;
      }

      rep->write_body("bin", seastar::sstring(res.data(), res.size()));
      return seastar::make_ready_future<HttpReplyPtr>(std::move(rep));
    });
  }

private:
  inline uint32_t GetDestShardId() {
    uint32_t dst_shard = local_shard_id_;
    if (available_request_units_ > 0) {
      --available_request_units_;
    } else {
      dst_shard = non_local_shard_ids_[nls_cursor_];
      nls_cursor_ = (nls_cursor_ + 1) % non_local_shard_ids_.size();
    }
    return dst_shard;
  }

private:
  std::vector<ServingActor_ref> actor_refs_;
  const uint32_t                local_shard_id_;
  const uint32_t                logging_period_;
  uint64_t                      num_processed_requests_{0};
  uint64_t                      start_time_{0};
  uint32_t                      nls_cursor_;
  std::vector<uint32_t>         non_local_shard_ids_;
  uint32_t                      available_request_units_;
  AdaptiveRateLimiter*          rate_limiter_;
};

EventHandler::EventHandler(WorkerId worker_id, uint16_t http_port,
                           AdaptiveRateLimiter* rate_limiter)
  : server_(), http_port_(http_port), worker_id_(worker_id),
    rate_limiter_(rate_limiter) {
}

seastar::future<> EventHandler::Start() {
  return server_.start().then([this] {
    return SetRoutes();
  }).then([this] {
    return server_.listen(http_port_);
  }).then([this] {
    LOG(INFO) << "EventHandler is listening on port "
              << http_port_ << " ......";
  });
}

seastar::future<> EventHandler::Stop() {
  return server_.stop();
}

seastar::future<> EventHandler::SetRoutes() {
  return server_.set_routes([this] (seastar::httpd::routes& r) {
    auto g_sid = act::GlobalShardIdAnchor();
    hiactor::scope_builder builder = hiactor::scope_builder(
      g_sid, MakeServingGroupScope());
    auto serving_ref = MakeServingActorInstRef(builder);
    const auto w_url = std::string("/serving/w") + std::to_string(worker_id_);
    r.add(seastar::httpd::operation_type::GET,
          seastar::httpd::url({w_url.data(), w_url.size()}),
          new RunQueryHandler(builder, rate_limiter_));
    return seastar::make_ready_future<>();
  });
}

}  // namespace dgs
