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

#ifndef GRAPHSCOPE_LOADER_CORE_BULK_LOADER_H_
#define GRAPHSCOPE_LOADER_CORE_BULK_LOADER_H_

#include "boost/asio.hpp"

#include "dataloader/batch_producer.h"
#include "dataloader/partitioner.h"

#include "checkpoint_parser.h"

namespace dgs {
namespace dataloader {
namespace gs {

class BulkLoadingThreadPool {
public:
  using Callback = std::function<void()>;

  static BulkLoadingThreadPool& GetInstance();

  /// Init and create worker threads.
  void Init();

  /// Join the worker threads.
  void Join();

  /// Stop and clear worker threads.
  void Finalize();

  /// Bulk load maxgraph store file records and write them into output queues.
  ///
  /// \param db_name db name of maxgraph store.
  /// \param schema_file maxgraph schema file.
  /// \param snapshot_id the snapshot id to checkout.
  /// \param callback callback after loading
  ///
  /// The ingesting func will be post to thread pool and executed by order.
  ///
  /// A maxgraph store handle will be created to scan vertices and edges in this
  /// store file.
  void Load(const BulkLoadingInfo& loading_info, Callback callback);

private:
  BulkLoadingThreadPool() = default;
  static void DoLoad(const BulkLoadingInfo& info, Callback callback);

private:
  std::unique_ptr<boost::asio::thread_pool> workers_;
};

inline
BulkLoadingThreadPool& BulkLoadingThreadPool::GetInstance() {
  static BulkLoadingThreadPool instance;
  return instance;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_CORE_BULK_LOADER_H_
