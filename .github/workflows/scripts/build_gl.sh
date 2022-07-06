#!/bin/bash

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
gl_root_dir=${script_dir}/../../../graphlearn
cores=$(cat < /proc/cpuinfo | grep -c "processor")

PYBIN=/opt/python/${PYABI}/bin
"${PYBIN}/pip" install numpy

pushd "${gl_root_dir}"

git submodule update --init
"${gl_root_dir}"/install_dependencies.sh

rm -rf built
rm -rf build

mkdir build
pushd build
cmake -DTESTING=OFF -DGL_PYTHON_BIN="${PYBIN}/python" ..
make python -j"${cores}"
popd

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

rm dist/*-linux*.whl

popd
