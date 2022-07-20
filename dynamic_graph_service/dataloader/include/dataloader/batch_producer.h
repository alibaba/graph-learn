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

#ifndef DATALOADER_BATCH_PRODUCER_H_
#define DATALOADER_BATCH_PRODUCER_H_

#include "cppkafka/utils/buffered_producer.h"

#include "dataloader/batch_builder.h"
#include "dataloader/options.h"

namespace dgs {
namespace dataloader {

/// The kafka message producer that write ready graph update batches into output kafka queues.
class BatchProducer {
public:
  BatchProducer();
  ~BatchProducer() = default;

  /// Synchronously produce a message from kafka message builder.
  void SyncProduce(const cppkafka::MessageBuilder& builder);

  /// Synchronously produce a message from batch builder.
  void SyncProduce(const BatchBuilder& builder);

private:
  cppkafka::BufferedProducer<std::string> producer_;
};

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_BATCH_PRODUCER_H_
