#!/bin/bash
#
# A script to perform unit tests for dynamic graph service.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
build_dir=${script_dir}/build
built_bin_dir=${script_dir}/built/bin

########################################################
# Deploy local kafka cluster
# Arguments:
#   None
########################################################
function deploy_kafka_cluster() {
  mkdir -p "${build_dir}"/kafka_cluster
  pushd "${build_dir}"/kafka_cluster
  if [ ! -f "kafka_2.13-3.0.0.tgz" ]; then
    wget -q https://graphlearn.oss-cn-hangzhou.aliyuncs.com/package/kafka_2.13-3.0.0.tgz
  fi
  if [ ! -d "kafka_2.13-3.0.0" ]; then
    tar -xvzf kafka_2.13-3.0.0.tgz
  fi
  popd

  pushd "${build_dir}"/kafka_cluster/kafka_2.13-3.0.0
  ./bin/zookeeper-server-start.sh config/zookeeper.properties &
  sleep 5s
  ./bin/kafka-server-start.sh config/server.properties &
  sleep 5s

  for topic in record-poller-ut-1 record-poller-ut-2 sampling-actor-ut sample-publisher-ut service-record-polling-ut service-sample-publishing-ut
  do
    ./bin/kafka-topics.sh --create --topic $topic --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
  done
  popd
}

########################################################
# Stop local kafka cluster and clear logs
# Arguments:
#   None
########################################################
function stop_kafka_cluster() {
  pushd "${build_dir}"/kafka_cluster/kafka_2.13-3.0.0
  ./bin/kafka-server-stop.sh config/server.properties
  sleep 2s
  ./bin/zookeeper-server-stop.sh config/zookeeper.properties
  sleep 2s
  rm -rf /tmp/kafka-logs
  rm -rf /tmp/zookeeper
  popd
}

########################################################
# Run unit tests
# Arguments:
#   None
########################################################
function run_ut() {
  pushd "${built_bin_dir}"
  # start naive coordinator first
  ./naive_coordinator 1 &
  # run tests
  for i in *_unittest
  do
    ./"$i"
  done
  popd
}

########################################################
# Make code coverage reports
# Arguments:
#   None
########################################################
function make_code_cov_reports() {
  pushd "${build_dir}"
  make code_coverage
  pushd "${build_dir}"/coverage
  lcov --capture --directory ./targets --output-file coverage.info
  lcov --remove coverage.info '/usr/include/*' '*third_party/*' -o coverage.info
  genhtml coverage.info --output-directory ./reports
  popd
  popd
}

deploy_kafka_cluster
run_ut
stop_kafka_cluster
make_code_cov_reports
