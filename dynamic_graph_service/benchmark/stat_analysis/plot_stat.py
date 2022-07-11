import numpy as np
import matplotlib.pyplot as plt

k_num_actor_shards = 36
k_top_n = 3
k_interval = 1000
k_ignore_index = 4

def PlotPollerStats():
  timestamps = []
  shard_ids = []

  with open("./poller_stats.txt") as f:
    lines = f.readlines()
    for line in lines:
      ts, sid = line.split(',')
      timestamps.append(np.uint64(ts))
      shard_ids.append(int(sid))

  tmp_min = np.min(timestamps)
  timestamps = timestamps - tmp_min

  # timeline max
  max_ts = np.max(timestamps)
  num_buckets = int(max_ts / float(k_interval)) + 1
  max_ts = num_buckets * k_interval

  collector = np.zeros((num_buckets, k_num_actor_shards))
  print("collector shape: ", collector.shape)

  assert(len(timestamps) == len(shard_ids))
  # populate data
  for i in range(len(timestamps)):
    bucket_id = int(timestamps[i] / k_interval)
    collector[bucket_id][shard_ids[i]] += 1

  # print(collector)

  length = collector.shape[0]
  xs = range(0, length)
  xs = [i + 0.5 for i in xs]

  sum_ys = []
  top_n_ys = []
  for i in range(length):
    arr = collector[i]
    sorted_index = np.argsort(arr)
    # print(arr[sorted_index[-k_top_n:]])
    y_topN = np.sum(arr[sorted_index[-k_top_n:]])
    y_sum = int(np.sum(arr))

    top_n_ys.append(y_topN)
    sum_ys.append(y_sum - y_topN)

  # print("top_n_ys: ", top_n_ys)
  # print("sum_ys: ", sum_ys)

  assert(len(xs) == len(sum_ys))
  assert(len(xs) == len(top_n_ys))

  fig = plt.figure(figsize=(25, 5))

  plt.bar(xs[k_ignore_index:], top_n_ys[k_ignore_index:], label="sum of top-{} shards".format(k_top_n))
  plt.bar(xs[k_ignore_index:], sum_ys[k_ignore_index:], bottom=top_n_ys[k_ignore_index:], label="sum of all shards except top-{} shards".format(k_top_n))

  plt.legend()

  plt.xlabel("Timeline (interval = {}ms)".format(k_interval))
  plt.ylabel("#record batches")

  plt.savefig("./poller_stats.png", dpi=200)
  # plt.show()


def PlotActorStats():
  xs = []
  ys = []

  with open("./actor_stats.txt") as f:
    lines = f.readlines()
    for line in lines:
      ts, pid = line.split(',')
      xs.append(np.uint64(ts))
      ys.append(int(pid))

  min_xs = np.min(xs)
  xs = xs - min_xs

  fig = plt.figure(figsize=(20, 5))

  plt.plot(xs, ys, '.', markersize=0.6, color="cornflowerblue")

  plt.yticks(range(0, 35, 2))
  plt.xticks(range(0, 1500000, 30000))

  plt.xlabel("Timestamp(ms)")
  plt.ylabel("Actor shard id")

  plt.xticks(color='w')

  plt.grid(axis="y", color='lightgray', linewidth=0.7)
  plt.grid(axis="x", color='lightgray', linewidth=0.7)

  plt.xticks()

  plt.savefig("actor_stats.png", dpi=200)

if __name__ == "__main__":
  PlotPollerStats()
  PlotActorStats()