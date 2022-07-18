# KNN
## Background
A large part of GNN applications is to use the output embedding results for vector recall, i.e., to use the generated embedding as a retrieval library from which to find the TopK highest matching vectors for a given input vector, i.e., K-nearest neighbor retrieval.

On the other hand, KNN is also an effective method to construct a graph. With known vectors, by calculating the similarity between vectors, the two vectors with higher similarity are considered to have some implied connection between them, i.e., an edge of the Graph is created.

For these considerations, GL supports large-scale distributed offline vector retrieval (KNN) as part of the GNN ecology. GL provides KNN-related APIs that return results in numpy format, which can be combined with other functions of GL, as well as with TF training, to give developers a complete experience.

## Usage
In the semantics of GL, all data are elements of graphs, and KNN's data source is no exception. In GL, we use Node of Graph as the source of KNN data, i.e., we support building Node data into a KNN retrieval library. Node needs to have float type attributes, and these attributes will be treated as vectors. Since GL naturally supports heterogeneous graphs, multiple KNN retrieval libraries can also exist, and the KNN retrieval libraries corresponding to different vertices are different.

For example, if Graph has vertices of type item, each with 21 float attributes, to create a KNN retrieval, the code would be as follows.
```python
import graphlearn as gl
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Task index")
flags.DEFINE_string("job_name", None, "worker or ps")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("tables", "", "odps table name")

if __name__ == "__main__":
  worker_count = len(FLAGS.worker_hosts.split(','))
  item_path = FLAGS.tables.split(',')
  gl.set_knn_metric(metric) # 0 is l2 distance(default), 1 is inner product.
  option = gl.IndexOption()
  option.name = "knn"
  option.index_type = "ivfflat"
  option.nlist = 5
  option.nprobe = 2
  g = gl.Graph() \
    .node(item_path, "item", decoder=gl.Decoder(attr_types=["float"] * 21), option=option) \
    .init(task_index=FLAGS.task_index, task_count=worker_count)

  inputs = np.array([[5.0] * 21, [31.0] * 21], dtype=np.float32)
  ids, distances = g.search("item", inputs, gl.KnnOption(k=3))
  print(ids, distances)

  g.close()
```
In this example, lines 21~23 add the vertex data and initialization of the graph, please refer to the chapter on graph objects for details. For the item type Node, an `option` option is added, which indicates additional operations to be performed on the Node. Here we pass a `gl.IndexOption()` object, which indicates the index to be established on the Node, as shown in lines 16-21, the index name is `knn` (must be knn), and the index type to establish knn is `ivfflat`. Line 27 is the constructed input data, 2 21-dimensional float32 vectors, note that it must be float32 and the dimensionality is consistent with the number of float attributes of the retrieval library. The input supports batch, i.e., the shape can be a numpy array of [N, 21].

Line 29 is the retrieval operation. The first argument of `g.search()` is the vertex type, the second argument is the input data, and the third argument is KnnOption, where only the value of k needs to be specified. The return result is two two-dimensional vectors of ids and distances, shape is [N, k], ids is the first k ids closest to the vector of inputs, int64 type, distances is the corresponding distance, float32 type.

Regarding index_type, besides `ivfflat`, `flat`, `ivfpq` are also supported, and the GPU version also supports `gpu_ivfflat`, `gpu_flat`, `gpu_ivfpq`. Among them, `ivfflat` is to build index after clustering, balancing accuracy and search cost, most commonly used; `flat` is strict search, guaranteeing correct but time consuming; `ivfpq` will compress accuracy, for example, compress 32 dimensions to 8 dimensions, reducing storage but losing accuracy. `option.nlist` and `option.nprobe` are the number of clusters for clustering and the number of clusters for retrieval, e.g. copolymerize 5 classes, only the 2 closest classes to the input vector are considered for retrieval. These two options do not work when index_type is flat. When index_type is ivfpq, the `option.m` parameter specifies the dimension of the compression, and note that option.m must be a factor of the number of float attributes.

`gl.set_knn_metric` is used to set the knn distance calculation, 0 is the l2 distance, 1 is the inner product distance, default 0.

An example of using KNN to evaluate recall can be found in examples/eval.