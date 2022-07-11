#!/bin/bash
set -x

ps -ef | grep "com.alibaba.maxgraph.servers.MaxNode" | grep -v grep | awk '{print $2}' | xargs kill -9