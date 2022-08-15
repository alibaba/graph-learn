# Tutorial

This document is an e2e tutorial of offline training and online inference for a GNN model with:
  - GraphLearn-Training(gl) for offline training.
  - Dynamic-Graph-Service(dgs) for online inference.
  - Tensorflow model serving.

Here is an example of a supervised job with EgoBipartiteSage, containing the following sections.
1. Prepare u2i and i2i bipartite graph data, including bulk-loading data for offline training and streaming data for online inference.
2. Train the EgoBipartiteSage model using the offline bulk-loading data, it contains user model and item model.
3. Exporting user model.
4. Deploying the model on tensorflow model serving.
5. Deploy the online dynamic graph service and ingest the streaming data.
6. Start Java Client, sample and predict.

The first 3 parts need to use the training framework GraphLearn-Training, a detailed description refs:[GraphLearn-Training](../gl/intro.md).

## 1. Prepare data

We provide a script to generate data with [u2i](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/conf/u2i/schema.u2i.json) schema.
The u2i graph contains two vertex type: `user` and `item`, and two edge type: `u2i`(from `user` to `item`) and `i2i`(from `item` to `item`).
Run this script:

```shell
cd dynamic_graph_service
python3 python/data/u2i/u2i_generator.py --output-dir /tmp/u2i_gen
```

This generator will first create a static graph for bulk-loading data, which is used for offline training.
While generating the streaming data, the previously generated bulk-loading data will be used as the initial state
of the graph whose timestamp are set to 0. After that, we generate more graph updates with increasing timestamps.
The generated data are stored in `/tmp/u2i_gen/training` and `/tmp/u2i_gen/streaming` respectively.


## 2. Train model offline

```shell
cd graphlearn/examples/tf/ego_bipartite_sage
python train.py
```

Ref to [GraphLearn-Training](../gl/intro.md) for more details.


## 3. Export TF SavedModel


First, export model as tf SavedModel, we need to filter some of the placeholders as model serving inputs based on the
computational graph, you can view the computational graph with the help of Tensorboard to determine the inputs for the
serving subgraph.

The offline training model is saved in `graphlearn/examples/tf/ego_bipartite_sage/ckpt`, and we save the final serving model in
`graphlearn/examples/tf/serving/ego_bipartite_sage` directory, the inputs to the user-subgraph are the placeholders `0,3,4` according to TensorBoard.

```shell
cd graphlearn/examples/tf/serving
python export_serving_model.py --input_ckpt_dir=../ego_bipartite_sage --input_ckpt_name=ckpt --placeholders=0,3,4 --output_model_path=./ego_bipartite_sage
```

Check inputs and output of saved model.
```shell
saved_model_cli show --dir ego_bipartite_sage/1/ --all
```

## 4. Deploy TF Model

Install tensorflow-model-server.

```shell
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
apt-get update && apt-get install tensorflow-model-server
```

Start tensorflow-model-server and deploy model

```shell
nohup tensorflow_model_server --port=9000  \
--model_name=egomodel   \
--model_base_path=graphlearn/examples/tf/serving/ego_bipartite_sage \
>server.log 2>&1
```

## 5. Deploy Dynamic Graph Service.

### 5.1 Deploy kafka service
The Dynamic-Graph-Service uses kafka queue service to store streaming graph updates and sampled results.

Deploy a simple local kafka cluster by:
```shell
wget https://graphlearn.oss-cn-hangzhou.aliyuncs.com/package/kafka_2.13-3.0.0.tgz
tar zxvf kafka_2.13-3.0.0.tgz
cd kafka_2.13-3.0.0

# start cluster
./bin/zookeeper-server-start.sh config/zookeeper.properties &
./bin/kafka-server-start.sh config/server.properties &

# create related topics used by dgs
./bin/kafka-topics.sh --create --topic record-batches --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
./bin/kafka-topics.sh --create --topic sample-batches --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
```

Some other helm charts can also be used to deploy a stable kafka service on k8s cluster,
e.g. [bitnami kafka](https://github.com/bitnami/charts/tree/master/bitnami/kafka).

### 5.2 Deploy dgs on k8s cluster
We provide a helm chart to deploy dgs, before this, make sure your k8s cluster has been created correctly and helm tools have been installed.
Besides, dgs uses a k8s ingress to expose its service, make sure your k8s cluster contains a nginx controller.

Get helm repo info first:
```shell
helm repo add dgs https://graphlearn.oss-cn-hangzhou.aliyuncs.com/charts/dgs/
helm repo update
```

Install the chart with release name `dgs-u2i`:
```shell
cd dynamic_graph_service
helm install dgs-u2i dgs/dgs \
    --set frontend.ingressHostName="dynamic-graph-service.info" \
    --set-file graphSchema=./conf/u2i/schema.u2i.json \
    --set kafka.dl2spl.brokers=[$your_kafka_cluster_ip:9092] \
    --set kafka.dl2spl.topic="record-batches" \
    --set kafka.dl2spl.partitions=4 \
    --set kafka.spl2srv.brokers=[$your_kafka_cluster_ip:9092] \
    --set kafka.spl2srv.topic="sample-batches" \
    --set kafka.spl2srv.partitions=4 \
    --set glog.toConsole=true
```

After installation, the helm notes will give you instructions to get the service ip, such as:
```shell
export DgsServiceIP=$(kubectl get ingress --namespace default dgs-u2i-frontend-ingress --output jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo $DgsServiceIP
```

Ref to [dgs deployment doc](deploy.md) for more details about service configuration.

### 5.3 Start dataloader
In order to access the dgs from your dataloader machine, you need to register its host name with its service ip first:
```shell
echo "$DgsServiceIP  dynamic-graph-service.info" >> /etc/hosts
```
Build dataloader first:
```shell
cd dynamic_graph_service/dataloader
mkdir build && cd build
cmake .. && make -j4
```
Use the default file dataloader to ingest streaming data file:
```shell
cd dynamic_graph_service/dataloader
./build/apps/file_loader/file_dataloader \
    --dgs-service-host dynamic-graph-service.info \
    --pattern-file /tmp/u2i_gen/streaming/u2i.pattern \
    --data-file /tmp/u2i_gen/streaming/u2i.streaming \
    --reversed-edges i2i \
    --batch-size 32 \
    --barrier u2i_finished
```
> **Tip**: If you want to deploy your dataloader in the same k8s cluster with dgs, you should make sure that the
> `externalTrafficPolicy` of nginx controller has been set to `Cluster`.


## 6. Sample and Predict

We provide a Java Client for Sampling from DGS and predict with TF model service.

```
cd dynamic_graph_service/gsl_client
mvn clean compile assembly:single

java -jar gsl_client-1.0-SNAPSHOT-jar-with-dependencies.jar http://dynamic-graph-service.info egomodel
```

Java command with the following args,
0: DGS service host name
1: Serving model name.

refs to `dynamic_graph_service/gsl_client/src/main/java/org/aliyun/App.java`
