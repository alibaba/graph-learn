# Tutorial

This document is a turotial of offline training and online inference for a GNN model with:
  - GraphLearn,
  - Dynamic graph service,
  - Tensorflow model serving.

Here is an example of a supervised job with EgoSage, containing the following sections.
1. Prepare data, including bulk-loading data for offline training and streaming data for online inference.
2. Train the EgoGage model using the offline bulk-loading data.
3. Exporting TF model.
4. Deploying TF model on tensorflow model serving.
4. Deploy the online graph sampling service.
5. Start Java Client, sample and predict.

## Prepare data
We start with cora as an example, online deployment with simulated data.
```shell
cd examples/data
python cora.py
```

TODO(@Seventeen17): use open streaming data source

## Train model offline
```shell
cd graphlearn/examples/tf/ego_sage
python train_supervised.py
```

## Export TF SavedModel
First, export model as tf SavedModel, we need to filter some of the placeholders as model serving inputs based on the computational graph, you can view the computational graph with the help of Tensorboard to determine the inputs for the serving subgraph.
The offline training model is saved in `graphlearn/examples/tf/ego_sage/ckpt`, and we save the final serving model in `./ego_sage_sup_model` directory, the inputs to the subgraphs are the placeholders `0,2,3`.

```shell
cd graphlearn/examples/tf/serving
python export_serving_model.py ../ego_sage ckpt ego_sage_sup_model 0,2,3
```

Check inputs and output of saved model.
```shell
saved_model_cli show --dir ego_sage_sup_model/1/ --all
```

Outputs are as following.
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

## Deploy TF Model
Install tensorflow-model-server.
```shell
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
apt-get update && apt-get install tensorflow-model-server
```
Start tensorflow-model-server and deploy model
```shell
nohup tensorflow_model_server --port=9000  \
--model_name=saved_model_modified   \
--model_base_path=/home/wenting.swt/code/graph-learn/examples/tf/ego_sage/saved_model_modified \
>server.log 2>&1
```

## Deploy Dynamic Graph Service.
Ref: k8s/README.md
TODO(@Seventeen17)

## Sample and Predict
We show a quick start example without dgs.
TODO(@Seventeen17): replace me when dgs deployment doc is ready.

```
cd dgs/gsl_client
mvn -Dtest=PredictClientTest test
```

The complete usgaes are shown in `App`.

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
