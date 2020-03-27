#!/usr/bin/env bash
for file in $(ls -ld $(find ./graphlearn/python))
do
  if [[ $file == */test_*.py ]]
  then
    echo $file
    python $file
  fi
done
