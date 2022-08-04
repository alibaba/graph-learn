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

#include "service/request/query_request.h"

namespace dgs {

InstallQueryRequest::InstallQueryRequest(act::BytesBuffer&& buf,
                                         bool is_chief)
  : buf_(std::move(buf)),
    rep_(GetMutableInstallQueryRequestRep(buf_.get_write())),
    is_chief_(static_cast<uint8_t>(is_chief)) {
}

InstallQueryRequest::InstallQueryRequest(InstallQueryRequest&& other)
  : buf_(std::move(other.buf_)),
    rep_(other.rep_),
    is_chief_(other.is_chief_) {
    other.rep_ = nullptr;
}

void InstallQueryRequest::dump_to(act::SerializableQueue& qu) {
  auto flag_buffer = act::BytesBuffer(
    reinterpret_cast<const char*>(&is_chief_), sizeof(uint8_t));
  qu.push(std::move(flag_buffer));
  qu.push(std::move(buf_));
}

InstallQueryRequest
InstallQueryRequest::load_from(act::SerializableQueue& qu) {
  auto flag_buffer = qu.pop();
  auto flag = *reinterpret_cast<const uint8_t*>(flag_buffer.get());
  return InstallQueryRequest{qu.pop(), static_cast<bool>(flag)};
}


UnInstallQueryRequest::UnInstallQueryRequest(act::BytesBuffer&& buf,
                                             bool is_chief)
  : buf_(std::move(buf)),
    rep_(GetMutableUnInstallQueryRequestRep(buf_.get_write())),
    is_chief_(static_cast<uint8_t>(is_chief)) {
}

UnInstallQueryRequest::UnInstallQueryRequest(UnInstallQueryRequest&& other)
  : buf_(std::move(other.buf_)),
    rep_(other.rep_),
    is_chief_(other.is_chief_) {
  other.rep_ = nullptr;
}


void UnInstallQueryRequest::dump_to(act::SerializableQueue& qu) {
  auto flag_buffer = act::BytesBuffer(
    reinterpret_cast<const char*>(&is_chief_), sizeof(uint8_t));
  qu.push(std::move(flag_buffer));
  qu.push(std::move(buf_));
}

UnInstallQueryRequest
UnInstallQueryRequest::load_from(act::SerializableQueue& qu) {
  auto flag_buffer = qu.pop();
  auto flag = *reinterpret_cast<const uint8_t*>(flag_buffer.get());
  return UnInstallQueryRequest{qu.pop(), static_cast<bool>(flag)};
}

RunQueryRequest::RunQueryRequest(RunQueryRequest&& other)
  : qid_(other.qid_), vid_(other.vid_) {
}

RunQueryRequest::RunQueryRequest(QueryId qid, VertexId vid)
  : qid_(qid), vid_(vid) {
}

void RunQueryRequest::dump_to(act::SerializableQueue& qu) {
}

RunQueryRequest RunQueryRequest::load_from(act::SerializableQueue& qu) {
  return RunQueryRequest{};
}

}  // namespace dgs
