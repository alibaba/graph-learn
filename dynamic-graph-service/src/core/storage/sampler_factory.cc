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

#include "core/storage/sampler_factory.h"

#include "common/log.h"

namespace dgs {
namespace storage {

std::unique_ptr<EdgeSampler> SamplerFactory::CreateEdgeSampler(
    SamplerType type, Capacity cap) {
  auto build_func = builder_directory_[type];
  std::unique_ptr<EdgeSampler> sampler_ptr = build_func(cap);
  return sampler_ptr;
}

void SamplerFactory::RegisterEdgeSampler(
    SamplerType type, EdgeSamplerInstBuilder func) {
  auto reserved_size = static_cast<size_t>(type + 1);
  if (reserved_size > builder_directory_.size()) {
    builder_directory_.resize(reserved_size);
  }
  if (builder_directory_[type]) {
    LOG(FATAL) << "Duplicate registered Sampler: " << type;
    return;
  }
  builder_directory_[type] = std::move(func);
}

}  // namespace storage
}  // namespace dgs
