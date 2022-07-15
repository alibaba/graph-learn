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

#ifndef DATALOADER_DATALOADER_H_
#define DATALOADER_DATALOADER_H_

#include "dataloader/batch_builder.h"
#include "dataloader/batch_producer.h"
#include "dataloader/partitioner.h"
#include "dataloader/schema.h"

namespace dgs {
namespace dataloader {

/// Initialing the dataloader components.
///
/// The initialization func will fetch the following init info from
/// dgs service and init the related data-loading components:
///   - Graph schema info
///   - Downstream kafka queues info
///   - Downstream partitioning info
///
/// This func must be called before your program runs.
///
void Initialize(const std::string& dgs_service_host);

/// Setting a barrier for current dataloader instance.
///
/// Sometimes, you may want to deploy your own data-loading cluster
/// with "dl_count" dataloader instances. To help users track the
/// data-loading progress of the cluster, we provide a barrier
/// mechanism to check the status of data produced from dataloader
/// to dgs service at a synchronized view.
///
/// A barrier is a global state and shared between all dataloader
/// instances in cluster. Setting a barrier will insert a checking-point
/// into the output data stream, when all produced data (produced from all
/// dataloader instances) before this checking-point are sampled and ready
/// for serving in dgs service, the barrier will be set to "ready" status.
///
/// A global barrier is uniquely identified by its "barrier_name". For a
/// specific barrier, you must set it on all dataloader instances separately,
/// along with current instance's "dl_id". A global barrier is invalid until
/// it is set on all dataloader instances.
///
/// On a gsl-client, you can check the barrier status or wait for it to be ready.
///
void SetBarrier(const std::string& dgs_service_host,
                const std::string& barrier_name,
                uint32_t dl_count,
                uint32_t dl_id);

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_DATALOADER_H_