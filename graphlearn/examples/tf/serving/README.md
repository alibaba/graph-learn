
`export_serving_model` is a utility script for export `SavedModel` for tensorflow model serving.
First, modify compute graph trained by GraphLearn tf models, replace
`IteratorGetNext` operator with placeholders. Then export the modified
graph as `SavedModel`.

Args:
  --input_ckpt_path: original checkpoint path.
  --input_ckpt: original checkpoint file name.
  --output_model_path: saved path for `SavedModel`.
  --export: value '' means not export SavedModel, else export.
  --placeholders: placeholder indexes that are inputs of serving model, string
    splited by \',\', each elem is int, for example, \'0,3,4\'. You can find the
    placehoder indcies according to tensorboard.
  --output_tensor: output tensor name, default as "output_embeddings".
  --version: model version, default is 1.
  Note,
  1) add the following example code for identifying the output of graph
      in serving,
    ``` python
    src_emb = model.forward(src_ego)
    output_embeddings = tf.identity(src_emb, name="output_embeddings")
    saver = tf.train.Saver()
    saver.save(sess, model_path)
    ```
  2) filter the placeholders in graph that are needed in sub-graph for serving

Example:

  ```
  python export_serving_model.py --input_ckpt_dir=../ego_bipartite_sage --input_ckpt_name=ckpt --export=''
  python export_serving_model.py --input_ckpt_dir=../ego_bipartite_sage --input_ckpt_name=ckpt --placeholders=0,3,4 --output_model_path=./ego_bipartite_sage
  ```