# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from setuptools import setup, Extension, find_packages
import shlex
from subprocess import check_output

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
CUT_PATH = sys.path[0]

extensions = []

include_dirs=[]
library_dirs=[]
libraries=[]
extra_compile_args=[]
extra_link_args=[]

include_dirs.append(ROOT_PATH)
include_dirs.append(ROOT_PATH + '/graphlearn/include')
include_dirs.append(ROOT_PATH + '/built')
include_dirs.append(ROOT_PATH + '/third_party/pybind11/pybind11/include')
include_dirs.append(ROOT_PATH + '/third_party/glog/build')
include_dirs.append(ROOT_PATH + '/third_party/protobuf/build/include')

library_dirs.append(ROOT_PATH + '/built/lib')

GXXGTEQ52 = os.popen("expr `g++ -dumpversion` '>=' 5.2").read()[0]
if GXXGTEQ52 == '0':
  extra_compile_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')

extra_compile_args.append('-D__USE_XOPEN2K8')
extra_compile_args.append('-std=gnu++11')
extra_compile_args.append('-fvisibility=hidden')
extra_link_args.append('-Wl,-rpath=$ORIGIN/python/lib/')


libraries.append('graphlearn_shared')

sources = [ROOT_PATH + '/graphlearn/python/py_export.cc']

graphlearn_extension = Extension(
    'pywrap_graphlearn',
    sources,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries)

extensions.append(graphlearn_extension)

GIT_BRANCH_NAME = check_output(shlex.split(
  'git rev-parse --abbrev-ref HEAD')).strip()
GIT_HEAD_REV = check_output(shlex.split('git rev-parse --short HEAD')).strip()
VERSION = check_output(shlex.split('cat {}/version.txt'.format(CUT_PATH))).strip()

setup(
    name='graphlearn',
    version=VERSION,
    description='Python Interface for Graph Neural Network',
    ext_package='graphlearn',
    ext_modules=extensions,
    packages=find_packages(exclude=["*.examples", "*.examples.*", "examples.*", "examples"]),
    package_dir={'graphlearn' : 'graphlearn'},
    package_data={'': ['python/lib/lib*.so*']},
    )
