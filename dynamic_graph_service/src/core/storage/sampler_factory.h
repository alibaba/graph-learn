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

#ifndef DGS_CORE_STORAGE_SAMPLER_FACTORY_H_
#define DGS_CORE_STORAGE_SAMPLER_FACTORY_H_

#include "common/sampler_type.h"
#include "core/io/record.h"
#include "core/storage/sampler.h"

namespace dgs {
namespace storage {

using EdgeSampler = Sampler<io::EdgeRecordView>;
using VertexSamplerPtr = std::unique_ptr<VertexSampler>;
using EdgeSamplerPtr = std::unique_ptr<EdgeSampler>;

typedef std::unique_ptr<EdgeSampler> (*EdgeSamplerInstBuilder)(Capacity cap);

class SamplerFactory {
  std::vector<EdgeSamplerInstBuilder> builder_directory_;
  SamplerFactory() = default;
public:
  static SamplerFactory& GetInstance();
  std::unique_ptr<EdgeSampler> CreateEdgeSampler(SamplerType type,
                                                 Capacity cap);
  void RegisterEdgeSampler(SamplerType type, EdgeSamplerInstBuilder func);
};

namespace registration {

template <typename T>
class EdgeSamplerRegistration {
public:
  explicit EdgeSamplerRegistration(SamplerType type) noexcept {
    static_assert(std::is_base_of<EdgeSampler, T>::value,
                  "T must be a derived class of EdgeSampler.");
    SamplerFactory::GetInstance().RegisterEdgeSampler(type, BuilderFunc);
  }
private:
  static std::unique_ptr<EdgeSampler> BuilderFunc(Capacity cap) {
    std::unique_ptr<T> ptr(new T(cap));
    return ptr;
  }
};

}  // namespace registration

inline
SamplerFactory& SamplerFactory::GetInstance() {
  static SamplerFactory instance_;
  return instance_;
}

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_SAMPLER_FACTORY_H_
