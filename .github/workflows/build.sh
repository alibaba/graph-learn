#!/bin/bash
set -e -x

apt-get update
apt-get -y install libssl-dev

PYBIN=/opt/python/${PYABI}/bin
"${PYBIN}/pip" install numpy

cd /io
git submodule update --init
make clean
make
make python PYTHON="${PYBIN}/python"

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

rm dist/*-linux*.whl