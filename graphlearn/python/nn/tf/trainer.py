# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import warnings
import tensorflow as tf

class Trainer(object):
  """ For training a model, we have to define tf.Session and tf.train.Server,
  and then configure them, as well as being aware of the steps and epochs
  during the training procedure. Usually, it is a complicated and repetitive
  work for algorithm developers. Here we use a Trainer object to simplify
  typical training procedures.

  Usage:
  ```
    # Define a Trainer object
    trainer = Trainer()
    # Define a DataFlow/BatchGraphFLow object
    df = tfg.DataFlow(query)
    # Construct model and loss
    u_emb, i_emb, loss = model_func()
  ```

  # Just train
  ```
    trainer.minimize(loss)
    trainer.step_to(df, 100)
    trainer.close()
  ```

  # Train and metric some intermediate results
  ```
    trainer.minimize(loss)

    def func(loss, u_emb, i_emb):
      print(ret_loss, metric(ret_u, ret_i))

    trainer.step_to_epochs(df, 2, [loss, u_emb, i_emb], func)
    trainer.close()
  ```

  # Dump the embeddings without updating the model
  ```
    def dump_func(u_emb, i_emb):
      writer.write(u_emb, i_emb)

    trainer.run_one_epoch(df, [u_emb, i_emb], dump_func)
    trainer.close()
  ```

  # Train with distributed cluster
  ```
    cluster = {"ps": ps_hosts, "worker": worker_hosts}
    trainer = Trainer(cluster_spec=cluster, job_name="worker", task_index=1)

    with trainer.context():
      u_emb, i_emb, loss = model_func()
    trainer.minimize(loss)
    trainer.step_to_epochs(df, 1)
    trainer.close()
  ```
  """

  def __init__(self,
               cluster_spec=None,
               job_name=None,
               task_index=None,
               optimizer=tf.train.AdamOptimizer(),
               ckpt_dir=None,
               **kwargs):
    self.cluster_spec = cluster_spec
    self.job_name = job_name
    self.task_index = task_index
    self.optimizer = optimizer
    self.ckpt_dir = ckpt_dir
    self.ckpt_freq = 1200 if ckpt_dir else None
    self.inited = False
    # steps and epochs used to record the number of trained steps and epochs.
    self._steps = 0
    self._epochs = 0

    if cluster_spec:
      os.environ["CLUSTER_SPEC"] = json.dumps(cluster_spec.as_dict())

  def __exit__(self, exc_type, exc_value, tracebac):
    self.close()
    if os.environ.get("CLUSTER_SPEC"):
      del os.environ["CLUSTER_SPEC"]
    return True

  @property
  def steps(self):
    return self._steps
  
  @property
  def epochs(self):
    return self._epochs

  def context(self):
    """ Only used when training in cluster. """
    device_setter = tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index,
        cluster=self.cluster_spec)
    return tf.device(device_setter)

  def minimize(self, loss, global_step=None):
    """ Optimize the given loss and return nothing. If ```global_step``` is
    None, the trainer object will create one by default.
    """
    if not global_step:
      self.global_step = tf.train.get_or_create_global_step()
    else:
      self.global_step = global_step

    self.train_op = self.optimizer.minimize(loss, self.global_step)

  def step(self, args=None, args_func=None, feed_dict=None):
    """ Do the optimization once which will try to minimize the loss. You can
    also track some intermediate results while training, such as loss and metric
    values.

    Args:
      args: A list, consists of the tensors you want to track. Default is None.
      args_func: A function to handle the returned results of ```args```.
      feed_dict: A TensorFlow feed_dict object.

    Return:
      If no args, nothing to return.
      If args and no args_func, return a list corresponding to ```args```.
      If args and args_func, return the result of ```args_func```.
    """
    if not self.inited:
      self._init()
    if not args:
      return self.sess.run(self.train_op, feed_dict=feed_dict)

    if isinstance(args, list) or isinstance(args, tuple):
      args = [self.train_op] + list(args)
    else:
      args = [self.train_op] + [args]

    rets = self.sess.run(args, feed_dict=feed_dict)
    if args_func:
      return args_func(rets[1:])
    else:
      return rets[1:]

  def run(self, args, args_func=None, feed_dict=None):
    """ Just return the results of ```args``` without updating the model.
    It is usually called when doing inference or validation.

    Args:
      args: A list, consists of the tensors you want to track. Default is None.
      args_func: A function to handle the returned results of ```args```.
      feed_dict: A TensorFlow feed_dict object.

    Return:
      If no args, nothing to return.
      If args and no args_func, return a list corresponding to ```args```.
      If args and args_func, return the result of ```args_func```.
    """
    if not self.inited:
      self._init()
    if not args:
      return None

    rets = self.sess.run(args, feed_dict=feed_dict)
    if args_func:
      return args_func(rets)
    else:
      return rets

  def step_to(self, dataflow, steps, args=None, args_func=None, feed_dict=None):
    """ Do ```self.step(args, args_func)``` for ```steps``` times.

    Args:
      dataflow: A DataFlow/BatchGrpahFlow object, which is used to track steps.
      steps: An integer, how many times to run ```self.step()```.

    Return:
      Nothing to return.
    """
    if not self.inited:
      self._init()
    self._init_iterator(dataflow.iterator)
    self._epochs = 0
    self._steps = 0
    for i in range(steps):
      try:
        self.step(args, args_func, feed_dict=feed_dict)
        self._steps = i
      except tf.errors.OutOfRangeError:
        self._init_iterator(dataflow.iterator)
        self._epochs += 1
        print('Epoch %d finished, steps at %d/%d.' % (self._epochs, i+1, steps))

  def step_to_epochs(self, dataflow, epochs, args=None, args_func=None, feed_dict=None):
    """ Do ```self.step(args, args_func)``` until
    ```tf.errors.OutOfRangeError``` occurs for ```epochs``` times.

    Args:
      dataflow: A DataFlow/BatchGrpahFlow object, which is used to track epochs.
      epochs: An integer, how many epochs to run ```self.step()```. An epoch is
      marked with finished once ```tf.errors.OutOfRangeError``` occurs.

    Return:
      Nothing to return.
    """
    if not self.inited:
      self._init()
    self._init_iterator(dataflow.iterator)
    self._epochs = 0
    self._steps = 0
    for i in range(epochs):
      try:
        while True:
          self.step(args, args_func, feed_dict=feed_dict)
          self._steps += 1
      except tf.errors.OutOfRangeError:
        self._init_iterator(dataflow.iterator)
        self._epochs += 1
        print('Epoch %d/%d finished.' % (i+1, epochs))

  def run_one_epoch(self, dataflow, args, args_func=None, feed_dict=None):
    """
    Args: 
      dataflow: A DataFlow/BatchGrpahFlow object, which is used to track epochs.
    """
    if not self.inited:
      self._init()
    self._init_iterator(dataflow.iterator)
    self._epochs = 0
    self._steps = 0
    try:
      while True:
        self.run(args, args_func, feed_dict=feed_dict)
        self._steps += 1
    except tf.errors.OutOfRangeError:
      self._epochs = 0
      print('Finished running a epoch.')

  def join(self):
    if self.cluster_spec and self.job_name == 'ps':
      self._build_server()
      self.server.join()

  def close(self):
    if self.inited:
      self.sess.close()
    self.inited = False

  def _init(self):
    if self.cluster_spec:
      self._build_cluster()
    else:
      self._build_local()
    self.inited = True

  def _init_iterator(self, iterator):
    if not iterator:
      warnings.warn("DataFlow does not contain iterator.")
      return
    if hasattr(self.sess, "_tf_sess"):
      self.sess._tf_sess().run(iterator.initializer)
    else:
      self.sess.run(iterator.initializer)

  def _build_local(self):
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    self.sess = tf.Session(config=conf)
    self.sess.run(tf.local_variables_initializer())
    self.sess.run(tf.global_variables_initializer())

  def _build_cluster(self):
    self._build_server()

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    self.sess = tf.train.MonitoredTrainingSession(
        master=self.server.target,
        is_chief=(self.task_index == 0),
        config=conf,
        checkpoint_dir=self.ckpt_dir,
        save_checkpoint_secs=self.ckpt_freq)

  def _build_server(self):
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    self.server = tf.train.Server(
        self.cluster_spec,
        job_name=self.job_name,
        task_index=self.task_index,
        config=conf)
