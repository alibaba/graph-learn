#!/bin/bash
#
# A script to perform unit tests for dynamic graph service.

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
build_dir=${script_dir}/../build
built_bin_dir=${script_dir}/../built/bin

########################################################
# Deploy local kafka cluster
# Arguments:
#   None
########################################################
function deploy_kafka_cluster() {
  mkdir -p "${script_dir}"/kafka
  rm -rf "${script_dir}"/kafka/*
  pushd "${script_dir}"/kafka
  wget -q https://graphlearn.oss-cn-hangzhou.aliyuncs.com/tool/kafka_2.13-3.0.0.tgz
  tar -xvzf kafka_2.13-3.0.0.tgz
  popd

  pushd "${script_dir}"/kafka/kafka_2.13-3.0.0
  ./bin/zookeeper-server-start.sh config/zookeeper.properties &
  ./bin/kafka-server-start.sh config/server.properties &
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
  pushd "${script_dir}"/kafka/kafka_2.13-3.0.0
  ./bin/kafka-server-stop.sh config/server.properties
  ./bin/zookeeper-server-stop.sh config/zookeeper.properties
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
function make_code_cov() {
  pushd "${build_dir}"
  make code_coverage
  lcov --capture --directory ./coverage --output-file coverage.info
  genhtml coverage.info --output-directory ../gcov-reports
  popd
}

deploy_kafka_cluster
run_ut
make_code_cov
stop_kafka_cluster
