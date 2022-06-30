/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_CONTRIB_KNN_INDEX_H_
#define GRAPHLEARN_CONTRIB_KNN_INDEX_H_

#include <cstdint>
#include <cstdlib>

namespace graphlearn {
namespace op {

class KnnIndex {
public:
  explicit KnnIndex(int32_t d) : ntotal_(0), d_(d), is_trained_(true) {}
  virtual ~KnnIndex() {}

  /// Perform training on a representative set of vectors.
  /// Some kind of KnnIndex may not need to be trained.
  /// @param n      batch size of training vectors
  /// @param data   training vecors, size n * d
  virtual void Train(size_t n, const float* data) = 0;

  /// Add n vectors of dimension d to the index.
  /// @param n      batch size of input matrix
  /// @param data   input matrix, size n * d
  /// @param ids    input labels, size n
  virtual void Add(size_t n, const float* data, const int64_t* ids) = 0;

  /// Query k nearest neighbors for n vectors of dimension d.
  /// Return at most k vectors. If there are not enough results for a
  /// query, the result array is padded with -1.
  /// @param n           batch size of input vectors
  /// @param inputs      input vectors to search, size n * d
  /// @param ids         output ids of the NNs, size n * k
  /// @param distances   output pairwise distances, size n * k
  virtual void Search(size_t n, const float* input, int32_t k,
                      int64_t* ids, float* distances) const = 0;

protected:
  size_t  ntotal_;
  int32_t d_;
  bool    is_trained_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_INDEX_H_
