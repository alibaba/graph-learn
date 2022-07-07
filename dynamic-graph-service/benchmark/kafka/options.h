/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_BENCHMARK_KAFKA_OPTIONS_H_
#define DGS_BENCHMARK_KAFKA_OPTIONS_H_

#include "boost/program_options.hpp"

namespace bpo = boost::program_options;

namespace benchmark {
namespace kafka {

enum WorkloadType {
  Producing,
  Consuming,
  Mixed,
};
extern WorkloadType workload_type;

extern std::string brokers;
extern std::string topic;
extern int32_t partition_num;

extern size_t message_size;
extern size_t message_num;

extern size_t linger_ms;
extern size_t batch_msg_num;

void SetOptions(const bpo::variables_map& map);

} // namespace kafka
} // namespace benchmark

#endif // DGS_BENCHMARK_KAFKA_OPTIONS_H_
