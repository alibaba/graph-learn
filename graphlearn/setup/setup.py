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

import numpy
import os
import sys
from setuptools import setup, Extension, find_packages
import shlex
from subprocess import check_output

# This version string should be updated when releasing a new version.
_VERSION = '1.0.1'

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
CUT_PATH = sys.path[0]
OPEN_KNN = os.getenv('OPEN_KNN')


extensions = []
include_dirs = []
library_dirs = []
libraries = []
extra_compile_args = []
extra_link_args = []

include_dirs.append('/usr/include')
include_dirs.append('/usr/local/include')
include_dirs.append(ROOT_PATH)
include_dirs.append(ROOT_PATH + '/src')
include_dirs.append(ROOT_PATH + '/src/include')
include_dirs.append(ROOT_PATH + '/../third_party/pybind11/pybind11/include')
include_dirs.append(ROOT_PATH + '/../third_party/glog/build/include')
include_dirs.append(ROOT_PATH + '/../third_party/grpc/build/include')
include_dirs.append(numpy.get_include())

library_dirs.append(ROOT_PATH + '/built/lib')
if OPEN_KNN == 'OPEN':
  extra_compile_args.append('-DOPEN_KNN')
extra_compile_args.append('-D__USE_XOPEN2K8')
extra_compile_args.append('-std=c++11')
extra_compile_args.append('-fvisibility=hidden')
if sys.platform == 'linux' or sys.platform == 'linux2':
  extra_link_args.append('-Wl,-rpath=$ORIGIN/python/lib/')

libraries.append('graphlearn_shared')
# if OPEN_KNN == 'OPEN':
#   libraries.append('knn_shared')

sources = [ROOT_PATH + '/python/c/py_export.cc',
           ROOT_PATH + '/python/c/py_client.cc']
if OPEN_KNN == 'OPEN':
  sources.append(ROOT_PATH + '/python/c/py_contrib.cc')

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

setup(
    name='graph-learn',
    version=_VERSION,
    description='Python Interface for Graph Neural Network',
    url="https://github.com/alibaba/graph-learn",
    project_urls={
      'Bug Tracker': 'https://github.com/alibaba/graph-learn/issues',
      'Documentation': 'https://graph-learn.readthedocs.io/en/latest/',
      'Source Code': 'https://github.com/alibaba/graph-learn',
    },
    ext_package='graphlearn',
    ext_modules=extensions,
    packages=find_packages(where='..'),
    package_dir={'graphlearn': '.'},
    package_data={'': ['python/lib/lib*.so*']},
    license='Apache License 2.0',
    )
