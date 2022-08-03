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

#include "actor/operator/batch_generator.h"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "actor/operator/op_ref_factory.h"

namespace graphlearn {
namespace actor {

// Remote Piece Batch Utils
const char* DelegateFetchFlag = "DelegateFetch";

inline std::vector<BatchLocation> GetRemotePieceLocations(
    const ShardDataInfoVecT& sorted_infos, unsigned batch_sz) {
  unsigned local_order = sorted_infos.size();
  for (unsigned i = 0; i < sorted_infos.size(); i++) {
    if (sorted_infos[i].shard_id == brane::local_shard_id()) {
      local_order = i;
    }
  }
  std::vector<BatchLocation> piece_locs;
  unsigned remaining = batch_sz;
  unsigned cursor = 0;
  for (unsigned i = 0; i < sorted_infos.size(); i++) {
    auto piece_len = sorted_infos[i].data_size % batch_sz;
    auto offset = sorted_infos[i].data_size - piece_len;
    if (remaining <= piece_len) {
      if (cursor == local_order) {
        piece_locs.emplace_back(sorted_infos[i].shard_id, offset, remaining);
      }
      piece_len -= remaining;
      offset += remaining;
      remaining = 0;
    }
    if (remaining == 0) {
      remaining = batch_sz;
      if (++cursor > local_order) break;
    }
    if (piece_len > 0) {
      if (cursor == local_order) {
        piece_locs.emplace_back(sorted_infos[i].shard_id, offset, piece_len);
      }
      remaining -= piece_len;
    }
  }
  return piece_locs;
}

inline unsigned SizeofPieceBatch(const std::vector<BatchLocation>& locs) {
  unsigned sz = 0;
  for (auto& loc : locs) {
    sz += loc.length;
  }
  return sz;
}

inline std::vector<BaseOperatorActorRef*>
GenerateDelegateRefs(const OpActorParams* params) {
  auto num_shards = brane::local_shard_count();
  std::vector<actor::BaseOperatorActorRef*> actor_refs;
  if (params) {
    actor_refs.reserve(num_shards);
    std::string op_name = params->node->OpName();
    ActorIdType op_actor_id = params->self_actor_id;
    for (int32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
      brane::scope_builder builder = brane::scope_builder(
        shard_id + brane::machine_info::sid_anchor());
      actor_refs.push_back(actor::OpRefFactory::Get().Create(
        op_name, op_actor_id, &builder));
    }
  }
  return actor_refs;
}

inline seastar::future<TensorMap> RemotePieceRequest(
    const BatchLocation& loc, BaseOperatorActorRef* ref) {
  TensorMap tm;
  ADD_TENSOR(tm.tensors_, DelegateFetchFlag, kInt64, 2);
  tm.tensors_[DelegateFetchFlag].AddInt64(loc.offset);
  tm.tensors_[DelegateFetchFlag].AddInt64(loc.length);
  return ref->Process(std::move(tm));
}

struct RemoteNodePieceCache {
public:
  RemoteNodePieceCache(const OpActorParams* params, const io::IdType* id_arr,
      std::vector<BatchLocation>&& location_list, unsigned piece_batch_sz)
    : refs_(GenerateDelegateRefs(params)), local_id_array_(id_arr),
      loc_list_(std::move(location_list)), cached_(false) {
    piece_cache_.reserve(piece_batch_sz);
  }

  ~RemoteNodePieceCache() {
    for (auto ref : refs_) {
      delete ref;
    }
  }

  inline seastar::future<> LoadPieceCache() {
    std::vector<seastar::future<TensorMap>> futs;
    for (auto &loc : loc_list_) {
      if (loc.shard_id == brane::local_shard_id()) {
        auto begin = local_id_array_ + loc.offset;
        piece_cache_.insert(piece_cache_.end(), begin, begin + loc.length);
      } else {
        futs.emplace_back(RemotePieceRequest(loc, refs_[loc.shard_id]));
      }
    }
    return seastar::when_all_succeed(futs.begin(), futs.end()).then(
        [this] (std::vector<TensorMap> results) {
      for (int i = 0; i < results.size(); ++i) {
        for (auto &tn : results[i].tensors_) {
          auto size = tn.second.Size();
          const int64_t* begin = tn.second.GetInt64();
          piece_cache_.insert(piece_cache_.end(), begin, begin + size);
        }
      }
      cached_ = true;
      return seastar::make_ready_future<>();
    });
  }

  inline bool Cached() const {
    return cached_;
  }

  inline const io::IdType* Data() const {
    return piece_cache_.data();
  }

  inline unsigned Size() const {
    return piece_cache_.size();
  }

private:
  std::vector<BatchLocation>         loc_list_;
  std::vector<BaseOperatorActorRef*> refs_;
  const io::IdType*                  local_id_array_;
  std::vector<io::IdType>            piece_cache_;
  bool                               cached_;
};

struct RemoteEdgePieceCache {
public:
  RemoteEdgePieceCache(const OpActorParams* params,
      io::GraphStorage* local_edge_storage,
      std::vector<BatchLocation>&& location_list, unsigned piece_batch_sz)
    : refs_(GenerateDelegateRefs(params)),
      local_edge_store_(local_edge_storage),
      loc_list_(std::move(location_list)), cached_(false) {
    src_cache_.reserve(piece_batch_sz);
    dst_cache_.reserve(piece_batch_sz);
    edge_cache_.reserve(piece_batch_sz);
  }

  ~RemoteEdgePieceCache() {
    for (auto ref : refs_) {
      delete ref;
    }
  }

  inline seastar::future<> LoadPieceCache() {
    std::vector<seastar::future<TensorMap>> futs;
    for (auto &loc : loc_list_) {
      if (loc.shard_id == brane::local_shard_id()) {
        for (io::IdType id = loc.offset; id < loc.offset + loc.length; id++) {
          edge_cache_.push_back(id);
          src_cache_.push_back(local_edge_store_->GetSrcId(id));
          dst_cache_.push_back(local_edge_store_->GetDstId(id));
        }
      } else {
        futs.emplace_back(RemotePieceRequest(loc, refs_[loc.shard_id]));
      }
    }
    return seastar::when_all_succeed(futs.begin(), futs.end()).then(
        [this] (std::vector<TensorMap> results) {
      for (int i = 0; i < results.size(); ++i) {
        CachePieceTensorMap(&results[i]);
      }
      cached_ = true;
      return seastar::make_ready_future<>();
    });
  }

  inline bool Cached() const {
    return cached_;
  }

  inline const io::IdType* SrcData() const {
    return src_cache_.data();
  }

  inline const io::IdType* DstData() const {
    return dst_cache_.data();
  }

  inline const io::IdType* EdgeData() const {
    return edge_cache_.data();
  }

  inline unsigned Size() const {
    return edge_cache_.size();
  }

private:
  inline void CachePieceTensorMap(TensorMap* tm) {
    auto* src_ptr = tm->tensors_[kSrcIds].GetInt64();
    auto src_len = tm->tensors_[kSrcIds].Size();
    src_cache_.insert(src_cache_.end(), src_ptr, src_ptr + src_len);

    auto* dst_ptr = tm->tensors_[kDstIds].GetInt64();
    auto dst_len = tm->tensors_[kDstIds].Size();
    dst_cache_.insert(dst_cache_.end(), dst_ptr, dst_ptr + dst_len);

    auto* edge_ptr = tm->tensors_[kEdgeIds].GetInt64();
    auto edge_len = tm->tensors_[kEdgeIds].Size();
    edge_cache_.insert(edge_cache_.end(), edge_ptr, edge_ptr + edge_len);
  }

private:
  std::vector<BatchLocation>         loc_list_;
  std::vector<BaseOperatorActorRef*> refs_;
  io::GraphStorage*                  local_edge_store_;
  std::vector<io::IdType>            src_cache_;
  std::vector<io::IdType>            dst_cache_;
  std::vector<io::IdType>            edge_cache_;
  bool                               cached_;
};

// Node Batch Generator Basic Utils
ShardDataInfoVecT
NodeBatchGenerator::GetSortedDataInfos(const std::string& type) {
  ShardDataInfoVecT data_info_vec;
  data_info_vec.reserve(brane::local_shard_count());
  for (uint32_t i = 0; i < brane::local_shard_count(); ++i) {
    auto noder = ShardedGraphStore::Get().OnShard(i)->GetNoder(type);
    auto shard_data_size = noder->GetLocalStorage()->Size();
    data_info_vec.emplace_back(shard_data_size, i);
  }
  std::sort(data_info_vec.begin(), data_info_vec.end(), DataSizeLess);
  return data_info_vec;
}

// Ordered/Shuffled Node Batch Generator
struct TraverseNodeBatchGenerator::Iterator {
public:
  Iterator(const io::IdType* id_arr,
      std::unique_ptr<RemoteNodePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : id_array_(id_arr), piece_(std::move(piece_cache)),
      batch_size_(batch_sz), intact_size_(intact_sz),
      total_size_(intact_size_ + piece_sz) {}

  virtual ~Iterator() = default;

  virtual seastar::future<TensorMap> Next() = 0;

protected:
  void AddTensorByOffset(TensorMap* tm, unsigned offset) {
    if (offset < intact_size_) {
      tm->tensors_[kNodeIds].AddInt64(id_array_[offset]);
    } else {
      tm->tensors_[kNodeIds].AddInt64(piece_->Data()[offset - intact_size_]);
    }
  }

protected:
  std::unique_ptr<RemoteNodePieceCache> piece_;
  const io::IdType* id_array_;
  const size_t      batch_size_;
  const size_t      intact_size_;
  const size_t      total_size_;
};

struct TraverseNodeBatchGenerator::OrderedIterator :
  public TraverseNodeBatchGenerator::Iterator {
public:
  OrderedIterator(const io::IdType* id_arr,
      std::unique_ptr<RemoteNodePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : Iterator(id_arr, std::move(piece_cache), batch_sz, intact_sz, piece_sz),
      cursor_(0) {}

  ~OrderedIterator() override = default;

  seastar::future<TensorMap> Next() override {
    if (cursor_ < intact_size_ || !piece_.get() || piece_->Cached()) {
      return InCacheNext();
    }
    return piece_->LoadPieceCache().then([this] {
      return InCacheNext();
    });
  }

private:
  seastar::future<TensorMap> InCacheNext() {
    auto len = std::min(total_size_ - cursor_, batch_size_);
    TensorMap tm;
    ADD_TENSOR(tm.tensors_, kNodeIds, kInt64, len);
    for (size_t i = 0; i < len; ++i) {
      AddTensorByOffset(&tm, cursor_);
      if (++cursor_ >= total_size_) {
        cursor_ = 0;
      }
    }
    return seastar::make_ready_future<TensorMap>(std::move(tm));
  }

private:
  size_t cursor_;
};

struct TraverseNodeBatchGenerator::ShuffledIterator
  : public TraverseNodeBatchGenerator::Iterator {
public:
  ShuffledIterator(const io::IdType* id_arr,
      std::unique_ptr<RemoteNodePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : Iterator(id_arr, std::move(piece_cache), batch_sz, intact_sz, piece_sz),
      cursor_(0), shuffle_offset_(0), rd_(), rng_(rd_())  {
    UpdateShuffleBuffer();
  }

  ~ShuffledIterator() override = default;

  seastar::future<TensorMap> Next() override {
    if (!piece_.get() || piece_->Cached()) {
      return InCacheShuffleNext();
    }
    return piece_->LoadPieceCache().then([this] {
      return InCacheShuffleNext();
    });
  }

private:
  seastar::future<TensorMap> InCacheShuffleNext() {
    auto len = std::min(shuffle_buffer_.size() - shuffle_offset_, batch_size_);
    TensorMap tm;
    ADD_TENSOR(tm.tensors_, kNodeIds, kInt64, len);
    for (size_t i = 0; i < len; ++i) {
      AddTensorByOffset(&tm, shuffle_buffer_[shuffle_offset_]);
      ++cursor_;
      if (++shuffle_offset_ == shuffle_buffer_.size()) {
        UpdateShuffleBuffer();
      }
    }
    return seastar::make_ready_future<TensorMap>(std::move(tm));
  }

  void UpdateShuffleBuffer() {
    shuffle_buffer_.clear();
    shuffle_offset_ = 0;
    if (cursor_ >= total_size_) {
      cursor_ = 0;
    }
    auto len = std::min(static_cast<size_t>(GLOBAL_FLAG(ShuffleBufferSize)),
                        total_size_ - cursor_);
    shuffle_buffer_.reserve(len);
    for (unsigned i = 0; i < len; ++i) {
      shuffle_buffer_.push_back(cursor_ + i);
    }
    std::shuffle(shuffle_buffer_.begin(), shuffle_buffer_.end(), rng_);
  }

private:
  size_t                     cursor_;
  std::vector<size_t>        shuffle_buffer_;
  size_t                     shuffle_offset_;
  std::random_device         rd_;
  std::default_random_engine rng_;
};

TraverseNodeBatchGenerator::TraverseNodeBatchGenerator(
      const std::string& type, unsigned batch_sz,
      const OpActorParams* params, const std::string& strategy)
    : NodeBatchGenerator(), iter_(nullptr) {
  auto* local_store = ShardedGraphStore::Get().
    OnShard(brane::local_shard_id())->GetNoder(type)->GetLocalStorage();
  unsigned local_data_size = local_store->Size();
  const io::IdType* local_id_array = local_store->GetIds()->data();
  auto local_intact_batch_num = local_data_size / batch_sz;

  auto sorted_data_infos = GetSortedDataInfos(type);
  auto piece_locs = GetRemotePieceLocations(sorted_data_infos, batch_sz);
  auto piece_batch_sz = SizeofPieceBatch(piece_locs);
  std::unique_ptr<RemoteNodePieceCache> piece_cache(nullptr);
  if (piece_batch_sz > 0) {
    piece_cache.reset(new RemoteNodePieceCache(params, local_id_array,
      std::move(piece_locs), piece_batch_sz));
  }
  if (strategy == "by_order") {
    iter_ = new OrderedIterator(local_id_array, std::move(piece_cache),
      batch_sz, local_intact_batch_num * batch_sz, piece_batch_sz);
  } else {
    iter_ = new ShuffledIterator(local_id_array, std::move(piece_cache),
      batch_sz, local_intact_batch_num * batch_sz, piece_batch_sz);
  }
}

TraverseNodeBatchGenerator::~TraverseNodeBatchGenerator() {
  delete iter_;
}

seastar::future<TensorMap> TraverseNodeBatchGenerator::NextBatch() {
  return iter_->Next();
}

// Random Node Batch Generator
RandomNodeBatchGenerator::RandomNodeBatchGenerator(
      const std::string& type, unsigned batch_size)
    : NodeBatchGenerator(), batch_size_(batch_size),
      rd_(), engine_(rd_()) {
  auto noder = ShardedGraphStore::Get().OnShard(
    brane::local_shard_id())->GetNoder(type);
  ids_ = noder->GetLocalStorage()->GetIds();
  data_size_ = noder->GetLocalStorage()->Size();
  dist_ = std::uniform_int_distribution<int32_t>(0, data_size_ - 1);
}

seastar::future<TensorMap> RandomNodeBatchGenerator::NextBatch() {
  TensorMap tm;
  ADD_TENSOR(tm.tensors_, kNodeIds, kInt64, batch_size_);

  for (unsigned i = 0; i < batch_size_; ++i) {
    int32_t rand = dist_(engine_);
    tm.tensors_[kNodeIds].AddInt64((*ids_)[rand]);
  }
  return seastar::make_ready_future<TensorMap>(std::move(tm));
}

// Edge Batch Generator Basic Utils
ShardDataInfoVecT
EdgeBatchGenerator::GetSortedDataInfos(const std::string& type) {
  ShardDataInfoVecT data_info_vec;
  data_info_vec.reserve(brane::local_shard_count());
  for (uint32_t i = 0; i < brane::local_shard_count(); ++i) {
    auto graph = ShardedGraphStore::Get().OnShard(i)->GetGraph(type);
    auto shard_data_size = graph->GetLocalStorage()->GetEdgeCount();
    data_info_vec.emplace_back(shard_data_size, i);
  }
  std::sort(data_info_vec.begin(), data_info_vec.end(), DataSizeLess);
  return data_info_vec;
}

// Ordered/Shuffled Edge Batch Generator
struct TraverseEdgeBatchGenerator::Iterator {
public:
  Iterator(io::GraphStorage* local_edge_storage,
      std::unique_ptr<RemoteEdgePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : edge_store_(local_edge_storage), piece_(std::move(piece_cache)),
      batch_size_(batch_sz), intact_size_(intact_sz),
      total_size_(intact_size_ + piece_sz) {}

  virtual ~Iterator() = default;

  virtual seastar::future<TensorMap> Next() = 0;

protected:
  void AddTensorsByOffset(TensorMap* tm, unsigned offset) {
    if (offset < intact_size_) {
      io::IdType edge_id = offset;
      tm->tensors_[kEdgeIds].AddInt64(edge_id);
      tm->tensors_[kSrcIds].AddInt64(edge_store_->GetSrcId(edge_id));
      tm->tensors_[kDstIds].AddInt64(edge_store_->GetDstId(edge_id));
    } else {
      tm->tensors_[kEdgeIds].AddInt64(
        piece_->EdgeData()[offset - intact_size_]);
      tm->tensors_[kSrcIds].AddInt64(
        piece_->SrcData()[offset - intact_size_]);
      tm->tensors_[kDstIds].AddInt64(
        piece_->DstData()[offset - intact_size_]);
    }
  }

protected:
  std::unique_ptr<RemoteEdgePieceCache> piece_;
  io::GraphStorage* edge_store_;
  const size_t      batch_size_;
  const size_t      intact_size_;
  const size_t      total_size_;
};

struct TraverseEdgeBatchGenerator::OrderedIterator
  : public TraverseEdgeBatchGenerator::Iterator {
public:
  OrderedIterator(io::GraphStorage* local_edge_storage,
      std::unique_ptr<RemoteEdgePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : Iterator(local_edge_storage, std::move(piece_cache), batch_sz,
      intact_sz, piece_sz), cursor_(0) {}

  ~OrderedIterator() override = default;

  seastar::future<TensorMap> Next() override {
    if (cursor_ < intact_size_ || !piece_.get() || piece_->Cached()) {
      return InCacheNext();
    }
    return piece_->LoadPieceCache().then([this] {
      return InCacheNext();
    });
  }

private:
  seastar::future<TensorMap> InCacheNext() {
    auto len = std::min(total_size_ - cursor_, batch_size_);
    TensorMap tm;
    ADD_TENSOR(tm.tensors_, kEdgeIds, kInt64, len);
    ADD_TENSOR(tm.tensors_, kSrcIds, kInt64, len);
    ADD_TENSOR(tm.tensors_, kDstIds, kInt64, len);
    for (size_t i = 0; i < len; ++i) {
      AddTensorsByOffset(&tm, cursor_);
      if (++cursor_ >= total_size_) {
        cursor_ = 0;
      }
    }
    return seastar::make_ready_future<TensorMap>(std::move(tm));
  }

private:
  size_t cursor_;
};

struct TraverseEdgeBatchGenerator::ShuffledIterator
  : public TraverseEdgeBatchGenerator::Iterator {
public:
  ShuffledIterator(io::GraphStorage* local_edge_storage,
      std::unique_ptr<RemoteEdgePieceCache>&& piece_cache,
      unsigned batch_sz, unsigned intact_sz, unsigned piece_sz)
    : Iterator(local_edge_storage, std::move(piece_cache), batch_sz, intact_sz,
      piece_sz), cursor_(0), shuffle_offset_(0), rd_(), rng_(rd_()) {
    UpdateShuffleBuffer();
  }

  ~ShuffledIterator() override = default;

  seastar::future<TensorMap> Next() override {
    if (!piece_.get() || piece_->Cached()) {
      return InCacheShuffleNext();
    }
    return piece_->LoadPieceCache().then([this] {
      return InCacheShuffleNext();
    });
  }

private:
  seastar::future<TensorMap> InCacheShuffleNext() {
    auto len = std::min(shuffle_buffer_.size() - shuffle_offset_, batch_size_);
    TensorMap tm;
    ADD_TENSOR(tm.tensors_, kEdgeIds, kInt64, len);
    ADD_TENSOR(tm.tensors_, kSrcIds, kInt64, len);
    ADD_TENSOR(tm.tensors_, kDstIds, kInt64, len);
    for (size_t i = 0; i < len; ++i) {
      AddTensorsByOffset(&tm, shuffle_buffer_[shuffle_offset_]);
      ++cursor_;
      if (++shuffle_offset_ == shuffle_buffer_.size()) {
        UpdateShuffleBuffer();
      }
    }
    return seastar::make_ready_future<TensorMap>(std::move(tm));
  }

  void UpdateShuffleBuffer() {
    shuffle_buffer_.clear();
    shuffle_offset_ = 0;
    if (cursor_ >= total_size_) {
      cursor_ = 0;
    }
    auto len = std::min(static_cast<size_t>(GLOBAL_FLAG(ShuffleBufferSize)),
                        total_size_ - cursor_);
    shuffle_buffer_.reserve(len);
    for (unsigned i = 0; i < len; ++i) {
      shuffle_buffer_.push_back(cursor_ + i);
    }
    std::shuffle(shuffle_buffer_.begin(), shuffle_buffer_.end(), rng_);
  }

private:
  size_t                     cursor_;
  std::vector<size_t>        shuffle_buffer_;
  size_t                     shuffle_offset_;
  std::random_device         rd_;
  std::default_random_engine rng_;
};

TraverseEdgeBatchGenerator::TraverseEdgeBatchGenerator(
      const std::string& type, unsigned batch_sz,
      const OpActorParams* params, const std::string& strategy)
    : EdgeBatchGenerator(), iter_(nullptr) {
  io::GraphStorage* local_edge_storage = ShardedGraphStore::Get().
    OnShard(brane::local_shard_id())->GetGraph(type)->GetLocalStorage();
  unsigned local_data_size = local_edge_storage->GetEdgeCount();
  auto local_intact_batch_num = local_data_size / batch_sz;

  auto sorted_data_infos = GetSortedDataInfos(type);
  auto piece_locs = GetRemotePieceLocations(sorted_data_infos, batch_sz);
  auto piece_batch_sz = SizeofPieceBatch(piece_locs);
  std::unique_ptr<RemoteEdgePieceCache> piece_cache(nullptr);
  if (piece_batch_sz > 0) {
    piece_cache.reset(new RemoteEdgePieceCache(params, local_edge_storage,
      std::move(piece_locs), piece_batch_sz));
  }
  if (strategy == "by_order") {
    iter_ = new OrderedIterator(local_edge_storage, std::move(piece_cache),
      batch_sz, local_intact_batch_num * batch_sz, piece_batch_sz);
  } else {
    iter_ = new ShuffledIterator(local_edge_storage, std::move(piece_cache),
      batch_sz, local_intact_batch_num * batch_sz, piece_batch_sz);
  }
}

TraverseEdgeBatchGenerator::~TraverseEdgeBatchGenerator() {
  delete iter_;
}

seastar::future<TensorMap> TraverseEdgeBatchGenerator::NextBatch() {
  return iter_->Next();
}

// Random Edge Batch Generator
RandomEdgeBatchGenerator::RandomEdgeBatchGenerator(
      const std::string& type, unsigned batch_size)
    : EdgeBatchGenerator(), batch_size_(batch_size),
      rd_(), engine_(rd_()) {
  storage_ = ShardedGraphStore::Get().OnShard(
    brane::local_shard_id())->GetGraph(type)->GetLocalStorage();
  edge_count_ = storage_->GetEdgeCount();
  dist_  = std::uniform_int_distribution<::graphlearn::io::IdType>(
    0, edge_count_ - 1);
}

seastar::future<TensorMap> RandomEdgeBatchGenerator::NextBatch() {
  TensorMap tm;
  ADD_TENSOR(tm.tensors_, kEdgeIds, kInt64, batch_size_);
  ADD_TENSOR(tm.tensors_, kSrcIds, kInt64, batch_size_);
  ADD_TENSOR(tm.tensors_, kDstIds, kInt64, batch_size_);

  for (unsigned i = 0; i < batch_size_; ++i) {
    auto edge_id = dist_(engine_);
    tm.tensors_[kEdgeIds].AddInt64(edge_id);
    tm.tensors_[kSrcIds].AddInt64(storage_->GetSrcId(edge_id));
    tm.tensors_[kDstIds].AddInt64(storage_->GetDstId(edge_id));
  }

  return seastar::make_ready_future<TensorMap>(std::move(tm));
}

}  // namespace actor
}  // namespace graphlearn
