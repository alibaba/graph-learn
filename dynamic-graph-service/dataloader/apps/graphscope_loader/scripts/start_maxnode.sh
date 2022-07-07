#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
maxgraph_dir="${script_dir}/../deps/maxgraph"

zk_servers="localhost:2181"
zk_base_path="/maxgraph/test"
kafka_servers="localhost:9092"
kafka_topic="graph_store"
store_path="/tmp/maxgraph_data/store"
store_partitions=4

for i in "$@"; do
  case $i in
    --zk-servers=*)
      zk_servers="${i#*=}"
      shift
      ;;
    --zk-base-path=*)
      zk_base_path="${i#*=}"
      shift
      ;;
    --kafka-servers=*)
      kafka_servers="${i#*=}"
      shift
      ;;
    --kafka-topic=*)
      kafka_topic="${i#*=}"
      shift
      ;;
    --store-path=*)
      store_path="${i#*=}"
      shift
      ;;
    --store-partitions=*)
      store_partitions="${i#*=}"
      shift
      ;;
    -*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "-- zookeeper servers: ${zk_servers}"
echo "-- zookeeper base path: ${zk_base_path}"
echo "-- kafka servers: ${kafka_servers}"
echo "-- kafka topic: ${kafka_topic}"
echo "-- store partitions: ${store_partitions}"

rm -f /tmp/maxgraph.config && \
sed -e "s@LOG4RS_CONFIG@${maxgraph_dir}/conf/log4rs.yml@g" \
    -e "s@zk.connect.string=graph_env:2181@zk.connect.string=${zk_servers}@g" \
    -e "s@zk.base.path=/maxgraph/graph_test@zk.base.path=${zk_base_path}@g" \
    -e "s@kafka.servers=graph_env:9092@kafka.servers=${kafka_servers}@g" \
    -e "s@kafka.topic=graph_test@kafka.topic=${kafka_topic}@g" \
    -e "s@store.data.path=./data@store.data.path=${store_path}@g" \
    -e "s@partition.count=8@partition.count=${store_partitions}@g" \
    -e "s@backup.enable=false@backup.enable=true@g" \
    -e "s@log.recycle.enable=true@log.recycle.enable=false@g" \
    "${maxgraph_dir}"/conf/config.template > /tmp/maxgraph.config

LOG_NAME=maxnode MAXGRAPH_CONF_FILE=/tmp/maxgraph.config "${maxgraph_dir}"/bin/store_ctl.sh max_node_maxgraph &
