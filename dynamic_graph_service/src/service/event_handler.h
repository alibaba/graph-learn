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

#ifndef DGS_SERVICE_EVENT_HANDLER_H_
#define DGS_SERVICE_EVENT_HANDLER_H_

#include "seastar/http/httpd.hh"

#include "common/options.h"
#include "common/utils.h"
#include "service/request/service_request.h"

namespace dgs {

class AdaptiveRateLimiter;

class EventHandler {
public:
  EventHandler(WorkerId worker_id, uint16_t http_port,
               AdaptiveRateLimiter* rate_limiter);

  seastar::future<> Start();
  seastar::future<> Stop();

private:
  seastar::future<> SetRoutes();

private:
  seastar::httpd::http_server_control server_;
  const uint16_t                      http_port_;
  const WorkerId                      worker_id_;
  AdaptiveRateLimiter*                rate_limiter_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_EVENT_HANDLER_H_
