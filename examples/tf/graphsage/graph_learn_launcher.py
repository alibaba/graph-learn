import os
import sys
import subprocess as sp
import json

def gen_cluster_info():
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  print("TF_CONFIG=", tf_config_json)
  tf_config = json.loads(tf_config_json)
  cluster = tf_config.get("cluster", {})
  if cluster is None:
    print("TF_CONFIG cluster is empty")
    return

  ps_hosts = []
  worker_hosts = []
  node_list = []
  for key, value in cluster.items():
    if "ps" == key:
      ps_hosts = value
    elif "worker" == key:
      worker_hosts = value
    node_list.extend(value)

  task = tf_config.get("task", {})
  if task is None:
    print("TF_CONFIG task is empty")
    return

  task_index = task['index']
  job_name = task['type']
  return ps_hosts, worker_hosts, job_name, task_index

def run_tensorflow_job(tf_args):
  cmd_str = "python "
  cmd_str += " ".join(tf_args)
  print("run graph-learn command:", cmd_str)
  return sp.call(cmd_str, shell=True)


if __name__ == "__main__":
  tf_args = sys.argv[1:] 
  if "TF_CONFIG" in os.environ:
    ps_hosts, worker_hosts, job_name, task_index = gen_cluster_info()
    tf_args.append("--ps_hosts=" + ",".join(ps_hosts))
    tf_args.append("--worker_hosts=" + ",".join(worker_hosts))
    tf_args.append("--job_name=" + job_name)
    tf_args.append("--task_index=" + str(task_index))
    run_tensorflow_job(tf_args)
  else:
    print("no TF_CONFIG")
  
