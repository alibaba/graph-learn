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

#ifndef DGS_BENCHMARK_KAFKA_DRIVER_H_
#define DGS_BENCHMARK_KAFKA_DRIVER_H_

#include <thread>
#include "options.h"
#include "cppkafka/consumer.h"
#include "cppkafka/producer.h"

namespace benchmark {
namespace kafka {

class Producer {
public:
  explicit Producer(int32_t partition_id);

  void Init();
  void Run();
  void Join();

  int64_t Start() const {
    return start_;
  }

  int64_t End() const {
    return end_;
  }

private:
  void DoProduce();
  void Flush();

private:
  const int32_t partition_id_;
  std::unique_ptr<cppkafka::Producer> producer_;
  std::thread thread_;
  int64_t start_ = 0;
  int64_t end_ = 0;
};

class Consumer {
public:
  explicit Consumer(int32_t partition_id);

  void Init();
  void Run();
  void Join();

  int64_t Start() const {
    return start_;
  }

  int64_t End() const {
    return end_;
  }

private:
  void DoConsume();

private:
  const int32_t partition_id_;
  std::unique_ptr<cppkafka::Consumer> consumer_;
  std::thread thread_;
  int64_t start_ = 0;
  int64_t end_ = 0;
};

class Driver {
public:
  static void RunWorkload(const bpo::variables_map& map);
};

} // namespace kafka
} // namespace benchmark

#endif // DGS_BENCHMARK_KAFKA_DRIVER_H_
