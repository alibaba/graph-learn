import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import sys
import matplotlib.pyplot as plt

k_num_pollers = 12

def PlotKafkaConsumeRate():
  timestamps = []
  poller_ids = []

  with open("./poller_stats.txt") as f:
    lines = f.readlines()
    for line in lines:
      ts, sid = line.split(',')
      timestamps.append(np.uint64(ts))
      poller_ids.append(int(sid))

  min_ts = sys.maxsize
  print("min_ts value is ", min_ts)

  buckets = []
  for i in range(k_num_pollers):
    buckets.append([])

  for i in range(len(timestamps)):
    buckets[poller_ids[i]].append(timestamps[i])
    min_ts = min(min_ts, timestamps[i])

  print("min_ts is ", min_ts)
  for i in range(k_num_pollers):
    buckets[i] = buckets[i] - min_ts

  for i in range(k_num_pollers):
    assert(all(x<=y for x, y in zip(buckets[i], buckets[i][1:])))

  fig = plt.figure()

  for i in range(k_num_pollers):
    plt.plot(range(0, len(buckets[i])), buckets[i], ':', label="poller-{}".format(i))

  plt.legend()

  plt.xlabel("Index")
  plt.ylabel("Timestamp (ms)")

  plt.savefig("./kafka_consume_rate.png", dpi=200)

if __name__ == "__main__":
  PlotKafkaConsumeRate()
