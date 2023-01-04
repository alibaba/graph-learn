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

#ifndef GRAPHLEARN_INCLUDE_CONSTANTS_H_
#define GRAPHLEARN_INCLUDE_CONSTANTS_H_

namespace graphlearn {

extern const char* kPartitionKey;
extern const char* kOpName;
extern const char* kNodeType;
extern const char* kEdgeType;
extern const char* kType;
extern const char* kSrcType;
extern const char* kDstType;
extern const char* kSrcIds;
extern const char* kDstIds;
extern const char* kNodeIds;
extern const char* kEdgeIds;
extern const char* kNeighborCount;
extern const char* kNeighborIds;
extern const char* kBatchSize;
extern const char* kIsSparse;
extern const char* kStrategy;
extern const char* kDegreeKey;
extern const char* kWeightKey;
extern const char* kLabelKey;
extern const char* kIntAttrKey;
extern const char* kFloatAttrKey;
extern const char* kStringAttrKey;
extern const char* kSideInfo;
extern const char* kDirection;
extern const char* kSegmentIds;
extern const char* kNumSegments;
extern const char* kSegments;
extern const char* kDistances;
extern const char* kRowIndices;
extern const char* kColIndices;
extern const char* kSeedType;
extern const char* kNbrType;
extern const char* kCount;
extern const char* kBatchShare;
extern const char* kUnique;
extern const char* kIntCols;
extern const char* kIntProps;
extern const char* kFloatCols;
extern const char* kFloatProps;
extern const char* kStrCols;
extern const char* kStrProps;
extern const char* kFilterType;
extern const char* kFilterIds;
extern const char* kDegrees;
extern const char* kEpoch;
extern const char* kNodeFrom;
extern const char* kSparseIds;

enum SystemState {
  kBlank = 0,
  kStarted = 1,
  kInited = 2,
  kReady = 3,
  kStopped = 4
};

namespace io {

enum DataFormat {
  kDefault = 1,
  kWeighted = 2,
  kLabeled = 4,
  kAttributed = 8
};

enum Direction {
  kOrigin,
  kReversed
};

}  // namespace io

enum DataType {
  kInt32,
  kInt64,
  kFloat,
  kDouble,
  kString,
  kUnknown
};

enum DeployMode {
  kLocal = 0,
  kServer = 1,
  kWorker = 2
};

enum PartitionMode {
  kNoPartition = 0,
  kByHash = 1
};

enum PaddingMode {
  kReplicate = 0,
  kCircular = 1
};

enum TrackerMode {
  kRpc = 0,
  kFileSystem = 1
};

enum NodeFrom {
  kEdgeSrc,
  kEdgeDst,
  kNode
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_CONSTANTS_H_
