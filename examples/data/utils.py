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
"""Common tools used by data preprocess scripts.
Contains download, and extract.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import urllib2
import zipfile

def download(url, path):
  if os.path.exists(path):
    return
  print('download from ' + url + '...')
  response = urllib2.urlopen(url)
  with open(path, 'wb') as output:
    output.write(response.read())

def extract(file, target_dir):
  print('extract file to ' + target_dir)
  if os.path.exists(target_dir):
    print('dataset dir already exists.')
    return
  if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
    with tarfile.open(file, 'r') as tar_ref:
      tar_ref.extractall(target_dir)
  elif file.endswith('.zip'):
    with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall(target_dir)
  else:
    raise Exception('Unsupported compression format ' + file)