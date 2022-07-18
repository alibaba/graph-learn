# 训练推理教程

本文档帮助我们利用GraphLearn-Training、动态图采样服务和tensorflow model serving，开始一个GNN模型的离线训练和在线推理。

这里以EgoSage的有监督模型为例，包含以下部分：
1. 准备数据，包含离线训练的批数据和在线推理的流数据。
2. 使用离线批数据训练EgoGage模型。
3. 导出模型。
4. 部署模型到tensorflow model serving上。
5. 部署在线图采样服务。
6. 启动Java Client，进行采样并预测。

 其中，前3部分我们需要用到训练框架GraphLearn-Training，详细介绍参考[GraphLearn-Training](../gl/intro.md)。

## 1. Prepare data
我们先用cora数据为例；在线部署用模拟的数据。
```shell
cd examples/data
python cora.py
```

TODO：用一套数据。

## 2. Train model offline
```shell
cd graphlearn/examples/tf/ego_sage
python train_supervised.py
```

## 3. Export TF SavedModel
首先，需要将模型导出，这里我们需要根据计算图筛选一部分输入作为model serving的输入，可以借助Tensorboard查看计算图，以确定serving子图的输入。
离线训练模型保持在graphlearn/examples/tf/ego_sage/ckpt中，这里我们将最终serving的model保存在./ego_sage_sup_model目录下，子图的输入是0，2，3这几个placeholder。
```shell
cd graphlearn/examples/tf/serving
python export_serving_model.py ../ego_sage ckpt ego_sage_sup_model 0,2,3
```

查看saved model的输入输出以确认正确：
```shell
saved_model_cli show --dir ego_sage_sup_model/1/ --all
```
输出应该如下：
```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['predict_actions']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['IteratorGetNext_ph_input_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1433)
        name: IteratorGetNext_placeholder:0
    inputs['IteratorGetNext_ph_input_2'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1433)
        name: IteratorGetNext_placeholder_2:0
    inputs['IteratorGetNext_ph_input_3'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1433)
        name: IteratorGetNext_placeholder_3:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_embeddings'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 7)
        name: output_embeddings:0
  Method name is: tensorflow/serving/predict
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
--model_name=saved_model_modified   \
--model_base_path=/home/wenting.swt/code/graph-learn/examples/tf/ego_sage/saved_model_modified \
>server.log 2>&1
```

## 5. Deploy dgs

## 5. Sample and Predict
我们给了一个快速开始的例子：
```
cd dgs/gsl_client
mvn -Dtest=PredictClientTest test
```

完整的写法参考App：
```java
String server = "http://dynamic-graph-service.info";
Graph g = Graph.connect(server);

Query query = g.V("user").feed(source).properties(1).alias("seed")
    .outV("u2i").sample(15).by("topk_by_timestamp").properties(1).alias("hop1")
    .outV("i2i").sample(10).by("topk_by_timestamp").properties(1).alias("hop2")
    .values();
Status s = g.install(query);

Decoder decoder = new Decoder(g);
decoder.addFeatDesc("user",
                    new ArrayList<String>(Arrays.asList("float")),
                    new ArrayList<Integer>(Arrays.asList(128)));
decoder.addFeatDesc("item",
                    new ArrayList<String>(Arrays.asList("float")),
                    new ArrayList<Integer>(Arrays.asList(128)));
ArrayList<Integer> phs = new ArrayList<Integer>(Arrays.asList(0, 1, 2));

TFPredictClient client = new TFPredictClient(decoder, "localhost", 9000);
for (int i = 0; i < iters; ++i) {
  Value content = g.run(query);
  EgoGraph egoGraph = content.getEgoGraph("seed");
  client.predict("model", 1, egoGraph, phs);
}
```
