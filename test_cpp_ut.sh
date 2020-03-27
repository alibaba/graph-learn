#!/usr/bin/env bash
source ./env.sh
cd ./built/bin/
for i in `ls *_unittest`
  do ./$i
done
