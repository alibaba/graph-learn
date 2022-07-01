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

#ifndef DGS_SERVICE_REQUEST_SERVICE_REQUEST_H_
#define DGS_SERVICE_REQUEST_SERVICE_REQUEST_H_

#include "common/actor_wrapper.h"
#include "common/typedefs.h"
#include "generated/fbs/stop_service_generated.h"
#include "generated/fbs/sync_meta_generated.h"

namespace dgs {

class SyncMetaRequest {
public:
  explicit SyncMetaRequest(actor::BytesBuffer &&buf);
  SyncMetaRequest(const SyncMetaRequest&) = delete;
  SyncMetaRequest(SyncMetaRequest &&other);

  ~SyncMetaRequest() = default;

  // TODO(wenting.swt): design meta
  uint32_t meta() const {
    return rep_->meta();
  }

  actor::BytesBuffer CloneBuffer() const {
    return buf_.clone();
  }

  void dump_to(actor::SerializableQueue &qu) {  // NOLINT
    qu.push(std::move(buf_));
  }

  static SyncMetaRequest load_from(actor::SerializableQueue &qu) {  // NOLINT
    return SyncMetaRequest(qu.pop());
  }

private:
  actor::BytesBuffer  buf_;
  SyncMetaRequestRep* rep_;
};

class StopServiceRequest {
public:
  explicit StopServiceRequest(actor::BytesBuffer &&buf);
  StopServiceRequest(const StopServiceRequest&) = delete;
  StopServiceRequest(StopServiceRequest &&other);

  ~StopServiceRequest() = default;

  bool force() const {
    return rep_->force();
  }

  actor::BytesBuffer CloneBuffer() const {
    return buf_.clone();
  }

  void dump_to(actor::SerializableQueue &qu) {  // NOLINT
    qu.push(std::move(buf_));
  }

  static StopServiceRequest load_from(actor::SerializableQueue &qu) {  // NOLINT
    return StopServiceRequest(qu.pop());
  }

private:
  actor::BytesBuffer    buf_;
  StopServiceRequestRep *rep_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_REQUEST_SERVICE_REQUEST_H_
