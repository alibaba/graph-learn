# Loading From Your Data Source

By using kafka queues as the entry for streaming graph updates, the Dynamic-Graph-Service (abbreviated as dgs) can
decouple the different data sources and sampling workers.
Users can develop their own data loaders to ingest the source data and push graph updates into the output kafka queues
for further consuming.

## Rules to Follow

When processing source data in the dataloader, these rules should be followed:
- The graph schema from source data must be consistent with dgs.
- A produced kafka message must be a batch of graph updates, defined as a flatbuffers table [RecordBatchRep](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/fbs/record.fbs).
- Graph updates are partitioned for sampling workers, all records in one batch must have the same data partition id.

## Dataloader SDK

In practice, it is difficult for users to batch records with dgs graph schema, manage partitioning logic and
produce partitioned batches into kafka queues with themselves.
Thus, we provide a dataloader sdk (a c++ lib) to help users to do these things.


### Build

Build dataloader sdk with cmake:
```shell
$ cd dynamic_graph_service/dataloader
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$install_prefix ..
$ make && make install
```

use it in CMakeLists.txt:
```cmake
list (APPEND CMAKE_PREFIX_PATH $install_prefix)
find_package (DataLoader)
target_link_libraries (program_name
  PUBLIC DataLoader::dataloader)
```

### Initialization
The dataloader must be initialized before the program runs.
We provide a simple func to help do this, after deploying dgs, specify the dgs host name and
the sdk will fetch info from dgs and init automatically.
```c++
void Initialize(const std::string& dgs_host);
```

### Usage
The following example code shows how to develop a simple dataloader with our sdk tools:

```c++
#include "dataloader/dataloader.h"  // include the sdk header

using namespace dgs::dataloader;

int main() {
    // Init sdk
    std::string dgs_host = "dynamic-graph-service.info";
    Initialize(dgs_host);
    
    // Using a group producer to auto partition, batch and produce updates,
    uint32_t batch_size = 8; // max record number in one batch
    GroupProducer p(batch_size);
    
    // Using schema to query types.
    // user -> buy -> item
    auto& schema = Schema::Get();
    auto user_type = schema.GetVertexDefByName("user").Type();
    auto item_type = schema.GetVertexDefByName("item").Type();
    auto buy_type = schema.GetEdgeDefByName("buy").Type();
    
    // add "user" vertex with vid = 111.
    p.AddVertex(user_type, 111, {
        {schema.GetAttrDefByName("name").Type(), STRING, "Li Hua"},
        {schema.GetAttrDefByName("country").Type(), STRING, "China"}
    });
    
    // add "item" vertex with vid = 222.
    p.AddVertex(item_type, 222, {
        {schema.GetAttrDefByName("name").Type(), STRING, "Coca-Cola"},
        {schema.GetAttrDefByName("price").Type(), FLOAT32, ToBytes<float>(2.5)}
    });
    
    // add "buy" edge
    p.AddEdge(buy_type, user_type, item_type, 111, 222, {
        {schema.GetAttrDefByName("timestamp").Type(), INT64, ToBytes<int64_t>(20220701)}
    });
    
    // flush and produce
    p.FlushAll();
}
```

We provide a default file loader to load a data file according to the data format defined by users,
refer to [file-loader](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/dataloader/apps/file_loader).

## Data Loading Barrier

Sometimes, users may want to deploy their own data-loading cluster with multiple dataloader instances.
To help users track the data-loading progress of the cluster, we provide a barrier mechanism to check
the status of data produced from dataloader to dgs service at a synchronized view.

A barrier is a global state and shared between all dataloader instances in cluster.
Setting a barrier will insert a checking-point into the output data stream (kafka queues),
when all produced data (produced from all dataloader instances) before this checking-point are sampled and ready
for serving in dgs service, the barrier will be set to "ready" status.

A global barrier is uniquely identified by its "barrier_name". For a specific barrier, it must be set on all
dataloader instances separately along with current instance's unique id.
A global barrier is invalid until it is set on all dataloader instances.

We provide an `SetBarrier` func to help set a barrier on a specific dataloader instance.
```c++
void SetBarrier(const std::string& dgs_host,
                const std::string& barrier_name,
                uint32_t dl_count,
                uint32_t dl_id);
```

For example, assuming that you need to bulk load a huge table with 3 partitions and further load your streaming data source,
you may create a cluster with 3 data loaders to load them concurrently.
On each dataloader instance, set a barrier after bulk load finished:
```c++
    // On dataloader 0
 
    // bulk load ...
    
    SetBarrier("dynamic-graph-service.info", "bulk_load", 3, 0);

    // streaming load ...
 
--------

    // On dataloader 1

    // bulk load ...

    SetBarrier("dynamic-graph-service.info", "bulk_load", 3, 1);

    // streaming load ...

--------

    // On dataloader 2

    // bulk load ...

    SetBarrier("dynamic-graph-service.info", "bulk_load", 3, 2);

    // streaming load ...
```

We provide another `CheckBarrier` func to help users check the barrier status
after all dataloader instances have set the barrier.

```c++
    /// Status enum of global barrier.
    ///  "NOT_SET":   The barrier has not been set.
    ///  "PRODUCED":  All data before this barrier have been produced but not sampled.
    ///  "SAMPLED":   All data before this barrier have been sampled but not ready for serving.
    ///  "READY":     All data before this barrier have been sampled and ready for serving.
    ///
    enum BarrierStatus {
        NOT_SET,
        PRODUCED,
        SAMPLED,
        READY
    };

    /// Check a global barrier status
    ///
    BarrierStatus CheckBarrier(const std::string& dgs_host, const std::string& barrier_name);
```

Users can also check barrier status on their gsl-clients:
```java
    TODO(@Seventeen17): add the java example code here
```
