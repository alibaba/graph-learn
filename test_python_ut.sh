#!/usr/bin/env bash
for test_file in $(find ./graphlearn/python/ -name "test_*.py")
do
echo TESTING $test_file ...
python $test_file
done
