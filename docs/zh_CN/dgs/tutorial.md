# 训练推理教程

本文档帮助我们利用GraphLearn-Training、动态图采样服务和tensorflow model serving，开始一个GNN模型的离线训练和在线推理。

这里以EgoBipartitrSage的无监督模型为例，包含以下部分：
1. 准备u2i和i2i二部图数据数据，包含离线训练的批数据和在线推理的流数据。
2. 使用离线批数据训练EgoBipartitrSage模型，它包含user侧和item侧两个模型。
3. 导出user侧的模型。
4. 部署模型到tensorflow model serving上。
5. 部署在线图采样服务，流入在线数据。
6. 启动Java Client，进行采样并预测。

 其中，前3部分我们需要用到训练框架GraphLearn-Training，详细介绍参考[GraphLearn-Training](../gl/intro.md)。

## 1. Prepare data
我们使用自定义的数据生成器，生成离线和流式的u2i和i2i二部图数据，包括新增的边和顶点属性的更新，对应u2i[u2i](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/conf/u2i/schema.u2i.json)数据schema。

```shell
cd dynamic_graph_service
python3 python/data/u2i/u2i_generator.py --output-dir /tmp/u2i_gen
```

该生成器将首先创建一个用于批量加载数据的静态图，用于离线训练。
在生成流数据时，前面生成的批数据将被用作图的初始状态，其时间戳被设置为0。
之后，更多的带时间戳的图更新将流入在线图服务。

产生的数据分别存储在`/tmp/u2i_gen/training`和`/tmp/u2i_gen/streaming`。

## 2. Train model offline

```shell
cd graphlearn/examples/tf/ego_bipartite_sage
python train.py
```
参考 [GraphLearn-Training](../gl/intro.md) for more details.

## 3. Export TF SavedModel

首先，需要将模型导出，这里我们需要根据计算图筛选一部分输入作为model serving的输入，可以借助Tensorboard查看计算图，以确定serving子图的输入。

离线训练模型保存在`graphlearn/examples/tf/ego_bipartite_sage/ckpt`中，这里我们将最终serving的model保存在`graphlearn/examples/tf/serving/ego_bipartite_sage`目录下，根据TensorBoard分析，user侧子图的输入是0, 3, 4这几个placeholder。

```shell
cd graphlearn/examples/tf/serving
python export_serving_model.py --input_ckpt_dir=../ego_bipartite_sage --input_ckpt_name=ckpt --placeholders=0,3,4 --output_model_path=./ego_bipartite_sage
```

查看saved model的输入输出以确认正确：
```shell
saved_model_cli show --dir ego_bipartite_sage/1/ --all
```

## 4. Deploy TF Model
首先安装tensorflow-model-server

```shell
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
apt-get update && apt-get install tensorflow-model-server
```

启动tensorflow-model-server，并部署模型

```shell
nohup tensorflow_model_server --port=9000  \
--model_name=egomodel   \
--model_base_path=graphlearn/examples/tf/serving/ego_bipartite_sage \
>server.log 2>&1
```

## 5. Deploy Dynamic Graph Service.

### 5.1 Deploy kafka service
动态图服务使用kafka来存储图的流式更新和样本。

部署一个简单的本地kafka集群：

```
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

这里提供了一些helm charts用于在k8s集群上部署一个stable kafka service，
e.g. [bitnami kafka](https://github.com/bitnami/charts/tree/master/bitnami/kafka).

### 5.2 Deploy dgs on k8s cluster

我们提供了一个部署dgs的helm chart, 在部署前，先确认kafka集群已经正确部署，并安装了helm。

DGS使用k8s ingress来提供服务，确保k8s集群包含nginx controller。

1. 获取helm repo:
```
helm repo add dgs https://graphlearn.oss-cn-hangzhou.aliyuncs.com/charts/dgs/
helm repo update
```

2. 安装 chart, release name为`dgs-u2i`:

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

安装完成后，helm note将提供获取service ip的指导，如：

```shell
export DgsServiceIP=$(kubectl get ingress --namespace default dgs-u2i-frontend-ingress --output jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo $DgsServiceIP
```

更多的配置参考：[dgs deployment doc](deploy.md)。

### 5.3 Start dataloader

为了让Dataloader所在的机器能够访问DGS，你需要先注册DGS的host name：
```shell
echo "$DgsServiceIP  dynamic-graph-service.info" >> /etc/hosts
```

编译DataLoader:

```shell
cd dynamic_graph_service/dataloader
mkdir build && cd build
cmake .. && make -j4
```

使用默认的file dataloader从文件中获取流数据：

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

> **Tip**: 如你想将dataloader部署到dgs同一个k8s集群中,你需要确保
> nginx controller的`externalTrafficPolicy` 配置为 `Cluster`.

## 6. Sample and Predict

我们提供了一个Java Client，以从DGS获取样本，并向TF model serivce发送predict请求，获取预测结果：

```
cd dynamic_graph_service/gsl_client
mvn clean compile assembly:single

java -jar gsl_client-1.0-SNAPSHOT-jar-with-dependencies.jar http://dynamic-graph-service.info egomodel
```
Java命令的参数包含,
args[0]: DGS service host name
args[1]: Serving model name.

详细的代码在`dynamic_graph_service/gsl_client/src/main/java/org/aliyun/App.java`中。