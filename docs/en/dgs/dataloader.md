# Loading From Your Data Source

By using kafka queues as the entry for streaming graph updates, the Dynamic-Graph-Service can decouple the different
data sources and sampling workers.
you should develop your own dataloader to read you data source and produce the graph updates into output kafka queues
for further consuming.

## Rules to Follow

When you process you source data and produce them in your dataloader, you should follow these rules:
- The graph schema from your source data must be consistent with dgs.
- A produced kafka message must be a batch of graph updates, defined as a flatbuffers table [RecordBatchRep](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/fbs/record.fbs).
- Graph updates and partitioned for sampling workers, all records in one batch must have the same data partition id.

## Dataloader SDK

In practice, it is difficult for users to batch records with dgs graph schema, manage partitioning logic and
produce partitioned batches into kafka queues with themselves.
Thus, we provide a dataloader sdk (a c++ lib) to help users to do these things.


### Build

Build dataloader sdk with cmake:
```shell
$ cd dynamic_graph_service/dataloader
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$your_install_prefix ..
$ make && make install
```

use it in your CMakeLists.txt:
```cmake
list (APPEND CMAKE_PREFIX_PATH $your_install_prefix)
find_package (DataLoader)
target_link_libraries(your_project_name
  PUBLIC DataLoader::dataloader)
```

### Initialization
You must initialize the dataloader sdk components before your program runs.
We provide a simple func to help do this, after deploying Dynamic-Graph-Service, specify the dgs host name and
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

We provide a default file loader to load a data file according to the data format you define,
refer to [file-loader](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/dataloader/apps/file_loader).

## Data Loading Barrier

Sometimes, you may want to deploy your own data-loading cluster with multiple dataloader instances.
To help users track the data-loading progress of the cluster, we provide a barrier mechanism to check
the status of data produced from dataloader to dgs service at a synchronized view.

A barrier is a global state and shared between all dataloader instances in cluster.
Setting a barrier will insert a checking-point into the output data stream (kafka queues),
when all produced data (produced from all dataloader instances) before this checking-point are sampled and ready
for serving in dgs service, the barrier will be set to "ready" status.

A global barrier is uniquely identified by its "barrier_name". For a specific barrier, you must set it on all
dataloader instances separately along with current instance's unique id.
A global barrier is invalid until it is set on all dataloader instances.

We provide an `SetBarrier` func to help you set a barrier on a specific dataloader instance.
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

After all dataloader instances have set the barrier, you can further check the barrier status on a gsl-client.

```java
    // FIXME(@Seventeen17): change the example code here
    Boolean ready = Client.CheckBarrier("bulk_load");
```
