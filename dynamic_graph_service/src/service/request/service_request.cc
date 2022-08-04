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

#include "service/request/service_request.h"

namespace dgs {

SyncMetaRequest::SyncMetaRequest(act::BytesBuffer &&buf)
  : buf_(std::move(buf)),
    rep_(GetMutableSyncMetaRequestRep(buf_.get_write())) {
}

SyncMetaRequest::SyncMetaRequest(SyncMetaRequest &&other)
  : buf_(std::move(other.buf_)),
    rep_(other.rep_) {
  other.rep_ = nullptr;
}

StopServiceRequest::StopServiceRequest(act::BytesBuffer &&buf)
  : buf_(std::move(buf)),
    rep_(GetMutableStopServiceRequestRep(buf_.get_write())) {
}

StopServiceRequest::StopServiceRequest(StopServiceRequest &&other)
  : buf_(std::move(other.buf_)),
    rep_(other.rep_) {
  other.rep_ = nullptr;
}

}  // namespace dgs
