/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "core/storage/sample_store.h"

#include "rocksdb/memtablerep.h"
#include "rocksdb/utilities/options_util.h"

#include "common/actor_wrapper.h"
#include "common/log.h"
#include "common/options.h"
#include "core/storage/prefix_extractor.h"

#define LOG_RETURN_IF_BACKUP_NOT_OK(s)       \
  if (!s.ok()) {                             \
    LOG(ERROR) << s.ToString();              \
    return;                                  \
  }                                          \

#define LOG_RETURN_IF_STATUS_NOT_OK(s)  \
  if (!s.ok()) {                        \
    LOG(ERROR) << s.ToString();         \
    return false;                       \
  }                                     \

#define RETURN_IF_RESULT_LESS_THAN_ZERO(res) \
  if (res < 0) {                             \
    return res;                              \
  }                                          \

namespace dgs {
namespace storage {

SampleStore::SampleStore(
    const std::vector<PartitionId>& pids,
    Partitioner&& partitioner,
    const std::string& db_path,
    const std::string& backup_path,
    RdbEnv* rdb_env)
  : db_path_(db_path),
    backup_path_(backup_path) {
  Configure(rdb_env);
  bool ok = true;
  for (auto pid : pids) {
    StorePartitionBackupInfo bk_info(pid);
    ok &= AddPartitionedDB(bk_info);
  }
  SetPartitioner(std::move(partitioner));
}

SampleStore::SampleStore(
    const std::vector<StorePartitionBackupInfo>& part_bk_infos,
    Partitioner&& partitioner,
    const std::string& db_path,
    const std::string& backup_path,
    RdbEnv* rdb_env)
  : db_path_(db_path),
    backup_path_(backup_path) {
  Configure(rdb_env);
  bool ok = true;
  for (auto& bk_info : part_bk_infos) {
    ok &= AddPartitionedDB(bk_info);
  }
  SetPartitioner(std::move(partitioner));
}

SampleStore::~SampleStore() {
  bool ok = true;
  for (uint32_t i = 0; i < db_insts_.size(); ++i) {
    ok &= RemovePartitionedDB(i);
  }

  if (backup_env_ != rocksdb::Env::Default()) {
    delete backup_env_;
    backup_env_ = nullptr;
  }
}

void SampleStore::Configure(RdbEnv* rdb_env) {
  // TODO(@xmqin): using HDFS enabled Env instead.
  backup_env_ = rocksdb::Env::Default();

  auto& gl_opts = Options::GetInstance().GetSampleStoreOptions();

  rdb_options_.create_if_missing = true;
  rdb_options_.env = rdb_env;
  // Use prefix_extractor since we don't need total order iteration.
  // FIXME(@xmqin):
  // maybe replace with CappedPrefixTransform(FixedPrefixTransform).
  rdb_options_.prefix_extractor = std::make_shared<RocksdbPrefixExtractor>();
  rdb_options_.max_open_files = -1;
  rdb_options_.max_background_jobs = 4;
  rdb_options_.compression = rocksdb::kLZ4Compression;

  // TODO(@xmqin): introduce rate-limiter, limiting the rate of
  // compactions and flushes to smooth I/O operations

  if (gl_opts.in_memory_mode) {
    read_options_.verify_checksums = false;
    rdb_options_.allow_mmap_reads = true;
    rdb_options_.table_factory.reset(rocksdb::NewPlainTableFactory());
  } else {
    // TODO(@xmqin): configure bloom filter.
    rocksdb::BlockBasedTableOptions table_options;
    table_options.index_type = rocksdb::BlockBasedTableOptions::kHashSearch;
    table_options.cache_index_and_filter_blocks = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    // TODO(@xmqin): set num_shards_bit
    table_options.block_cache = rocksdb::NewLRUCache(
        gl_opts.block_cache_capacity);
    rdb_options_.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));
  }

  if (gl_opts.memtable_rep == "hashskiplist") {
    rdb_options_.memtable_factory.reset(rocksdb::NewHashLinkListRepFactory(
        gl_opts.hash_bucket_count));
    rdb_options_.allow_concurrent_memtable_write = false;
  } else if (gl_opts.memtable_rep == "skiplist") {
    rdb_options_.memtable_factory.reset(new rocksdb::SkipListFactory(
        gl_opts.skip_list_lookahead));
    rdb_options_.allow_concurrent_memtable_write = true;
  } else {
    LOG(FATAL) << "Unsupported memtable rep: " << gl_opts.memtable_rep;
  }

  // disable Write-Ahead-Log.
  write_options_.disableWAL = true;
  // only iterate in a prefix-same range.
  read_options_.prefix_same_as_start = true;

  // other configurations.
  local_part_num_ = 0;
  ttl_in_seconds_ = static_cast<int32_t>(gl_opts.ttl_in_hours) * 3600;
}

bool SampleStore::PutVertex(const Key& key,
                            const io::RecordSlice& value_slice) {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].vtx_db;
  const auto key_slice = key.ToSlice();
  auto s = db->Put(write_options_,
                   {key_slice.data(), key_slice.size()},
                   {value_slice.data(), value_slice.size()});
  return s.ok();
}

bool SampleStore::PutEdge(const Key& key,
                          const io::RecordSlice& value_slice) {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].edge_db;
  const auto key_slice = key.ToSlice();
  auto s = db->Put(write_options_,
                   {key_slice.data(), key_slice.size()},
                   {value_slice.data(), value_slice.size()});
  return s.ok();
}

bool SampleStore::GetVertex(const Key& key, io::Record* record) const {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].vtx_db;
  // TODO(@xmqin): use pinnable value when possible.
  std::string value;
  const auto slice = key.ToSlice();
  auto s = db->Get(read_options_, {slice.data(), slice.size()}, &value);
  if (s.ok()) {
    auto *data = const_cast<char*>(value.data());
    auto size = value.size();
    auto buf = actor::BytesBuffer(data, size,
        seastar::make_object_deleter(std::move(value)));
    *record = io::Record{std::move(buf)};
  }
  return s.ok();
}

bool SampleStore::GetEdge(const Key& key, io::Record* record) const {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].edge_db;
  // TODO(@xmqin): use pinnable value when possible.
  std::string value;
  const auto slice = key.ToSlice();
  auto s = db->Get(read_options_, {slice.data(), slice.size()}, &value);
  if (s.ok()) {
    auto *data = const_cast<char*>(value.data());
    auto size = value.size();
    auto buf = actor::BytesBuffer(data, size,
        seastar::make_object_deleter(std::move(value)));
    *record = io::Record{std::move(buf)};
  }
  return s.ok();
}

bool SampleStore::GetVerticesByPrefix(const Key::Prefix& prefix_key,
                                      std::vector<KVPair>* records) const {
  auto pid = partitioner_.GetPartitionId(prefix_key.vid);
  auto *db = db_insts_[pid].vtx_db;
  rocksdb::Iterator *iter = db->NewIterator(read_options_);
  const auto slice = prefix_key.ToSlice();
  iter->Seek(rocksdb::Slice{slice.data(), slice.size()});
  auto status = iter->Valid();

  while (iter->Valid()) {
    auto val_slice = iter->value();
    // auto buf = actor::BytesBuffer(val_slice.data(), val_slice.size());
    io::Record record({val_slice.data(), val_slice.size()});
    records->emplace_back(Key::FromSlice({iter->key().data(),
                          iter->key().size()}),
                          std::move(record));
    iter->Next();
  }
  delete iter;

  return status;
}

bool SampleStore::GetEdgesByPrefix(const Key::Prefix& prefix_key,
                                   std::vector<KVPair>* records) const {
  auto pid = partitioner_.GetPartitionId(prefix_key.vid);
  auto *db = db_insts_[pid].edge_db;
  rocksdb::Iterator *iter = db->NewIterator(read_options_);
  const auto slice = prefix_key.ToSlice();
  iter->Seek(rocksdb::Slice{slice.data(), slice.size()});
  auto status = iter->Valid();

  while (iter->Valid()) {
    auto val_slice = iter->value();
    io::Record record({val_slice.data(), val_slice.size()});
    records->emplace_back(Key::FromSlice({iter->key().data(),
                          iter->key().size()}),
                          std::move(record));
    iter->Next();
  }
  delete iter;

  return status;
}

bool SampleStore::DeleteVertex(const Key& key) {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].vtx_db;
  const auto slice = key.ToSlice();
  auto s = db->Delete(write_options_, {slice.data(), slice.size()});
  return s.ok();
}

bool SampleStore::DeleteEdge(const Key& key) {
  auto pid = partitioner_.GetPartitionId(key.pkey.vid);
  auto *db = db_insts_[pid].edge_db;
  const auto slice = key.ToSlice();
  auto s = db->Delete(write_options_, {slice.data(), slice.size()});
  return s.ok();
}

bool SampleStore::DeleteVerticesByPrefix(const Key::Prefix& prefix_key) {
  // TODO(@xmqin):
  return true;
}

bool SampleStore::DeleteEdgesByPrefix(const Key::Prefix& prefix_key) {
  // TODO(@xmqin):
  return true;
}

std::vector<StorePartitionBackupInfo> SampleStore::Backup() {
  // Note that it may take a long time for this function to return
  auto num_part = partitioner_.GetPartitionsNum();
  std::vector<StorePartitionBackupInfo> part_bk_infos;
  for (PartitionId pid = 0; pid < num_part; ++pid) {
    StorePartitionBackupInfo bk_info(pid, false);
    BackupPartitionedDB(bk_info);
    if (bk_info.valid) {
      part_bk_infos.emplace_back(bk_info);
    }
  }
  return part_bk_infos;
}

bool SampleStore::AddPartitionedDB(const StorePartitionBackupInfo& bk_info) {
  auto pid = bk_info.pid;
  bool is_restore = bk_info.valid;
  std::lock_guard<std::mutex> guard(db_insts_mtx_);
  if (pid < db_insts_.size() && db_insts_[pid].is_open) {
    LOG(WARNING) << "Storage Partition " << pid << " already exists."
                 << " DB path is " << db_path_;
    return true;
  }

  if (pid >= db_insts_.size()) {
    // expand vector size.
    db_insts_.resize(pid + 1);
  }

  PartitionedDB pdb;
  bool ok;
  if (is_restore) {
    LOG(INFO) << "Restoring Storage Partition " << pid
              << ". vertex db backup id: " << bk_info.vertex_bid
              << ", edge db backup id: " << bk_info.edge_bid;
    ok = pdb.Restore(bk_info, db_path_, backup_path_, rdb_options_,
                     backup_env_, ttl_in_seconds_);
  }
  else {
    ok = pdb.Open(pid, db_path_, backup_path_, rdb_options_,
                     backup_env_, ttl_in_seconds_);
  }

  if (ok) {
    db_insts_[pid] = std::move(pdb);
    ++local_part_num_;
    LOG(INFO) << "Storage Partition " << pid << " is added."
              << " DB path is " << db_path_;
  } else {
    LOG(ERROR) << "Storage Partition " << pid << " add failed."
               << " DB path is " << db_path_;
  }
  return ok;
}

bool SampleStore::RemovePartitionedDB(PartitionId pid) {
  std::lock_guard<std::mutex> guard(db_insts_mtx_);
  if (pid >= db_insts_.size() || !db_insts_[pid].is_open) {
    LOG(WARNING) << "Storage Partition " << pid << " doesn't exist."
                 << " DB path is " << db_path_;
    return true;
  }

  bool ok = db_insts_[pid].Close();
  if (ok) {
    db_insts_[pid] = PartitionedDB();
    --local_part_num_;
  } else {
    LOG(ERROR) << "Storage Partition " << pid << " Close() failed."
               << " DB path is " << db_path_;
  }
  return ok;
}

void SampleStore::BackupPartitionedDB(StorePartitionBackupInfo& bk_info) {
  auto pid = bk_info.pid;
  if (pid >= db_insts_.size() || !db_insts_[pid].is_open) {
    // This partition is not belong to current worker, skip.
    return;
  }

  rocksdb::IOStatus s;
  rocksdb::BackupID backup_id;
  rocksdb::CreateBackupOptions options;
  // Since write-ahead logs are disabled
  options.flush_before_backup = true;

  // FIXME(@xmqin): should we add a is_backing_up flag?

  auto &pdb = db_insts_[pid];
  // backup vertex store.
  s = pdb.vtx_be->CreateNewBackup(options, pdb.vtx_db, &backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);
  s = pdb.vtx_be->VerifyBackup(backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);

  bk_info.vertex_bid = backup_id;

  // backup edge store.
  s = pdb.edge_be->CreateNewBackup(options, pdb.edge_db, &backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);
  s = pdb.edge_be->VerifyBackup(backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);

  bk_info.edge_bid = backup_id;
  bk_info.valid = true;
}

void SampleStore::SetPartitioner(Partitioner&& partitioner) {
  LOG(INFO) << "New partitioner is set."
            << " DB path is " << db_path_;
  partitioner_ = std::move(partitioner);
}

size_t SampleStore::GetLocalPartitionNum() const {
  return local_part_num_;
}

SampleStore::PartitionedDB::PartitionedDB(PartitionedDB&& other)
  : vtx_db(other.vtx_db), edge_db(other.edge_db),
    vtx_be(other.vtx_be), edge_be(other.edge_be),
    is_open(other.is_open) {
  other.vtx_db = nullptr;
  other.edge_db = nullptr;
  other.vtx_be = nullptr;
  other.edge_be = nullptr;
  other.is_open = false;
}

SampleStore::PartitionedDB&
SampleStore::PartitionedDB::operator=(PartitionedDB&& other) {
  if (this != &other) {
    vtx_db = other.vtx_db;
    edge_db = other.edge_db;
    vtx_be = other.vtx_be;
    edge_be = other.edge_be;
    is_open = other.is_open;

    other.vtx_db = nullptr;
    other.edge_db = nullptr;
    other.vtx_be = nullptr;
    other.edge_be = nullptr;
    other.is_open = false;
  }
  return *this;
}

bool SampleStore::PartitionedDB::Open(
    PartitionId partition_id,
    const std::string& db_path,
    const std::string& backup_path,
    const rocksdb::Options& options,
    rocksdb::Env* backup_env,
    uint32_t ttl_in_seconds) {
  // TODO(@xmqin): check whether db_path&backup_path is end with '/'
  auto vdb_name = db_path + "/vstore_part_" + std::to_string(partition_id);
  auto vbe_name = backup_path + "/vstore_bak_part_"
                              + std::to_string(partition_id);
  auto edb_name = db_path + "/estore_part_" + std::to_string(partition_id);
  auto ebe_name = backup_path + "/estore_bak_part_"
                              + std::to_string(partition_id);

  rocksdb::Status s;
  rocksdb::IOStatus io_s;
  // Open vertex db and its backup engine.
  s = rocksdb::DB::Open(options, vdb_name, &vtx_db);
  LOG_RETURN_IF_STATUS_NOT_OK(s);
  io_s = rocksdb::BackupEngine::Open(
      backup_env, rocksdb::BackupEngineOptions(vbe_name), &vtx_be);
  LOG_RETURN_IF_STATUS_NOT_OK(io_s);

  // Open edge db and its backup engine.
  s = rocksdb::DBWithTTL::Open(options, edb_name, &edge_db, ttl_in_seconds);
  LOG_RETURN_IF_STATUS_NOT_OK(s);

  io_s = rocksdb::BackupEngine::Open(
      backup_env, rocksdb::BackupEngineOptions(ebe_name), &edge_be);
  LOG_RETURN_IF_STATUS_NOT_OK(io_s);

  is_open = true;
  return true;
}

bool SampleStore::PartitionedDB::Restore(
    const StorePartitionBackupInfo& bk_info,
    const std::string& db_path,
    const std::string& backup_path,
    const rocksdb::Options& options,
    rocksdb::Env* backup_env,
    uint32_t ttl_in_seconds) {
  auto partition_id = bk_info.pid;
  auto vdb_name = db_path + "/vstore_part_" + std::to_string(partition_id);
  auto vbe_name = backup_path + "/vstore_bak_part_"
                              + std::to_string(partition_id);
  auto edb_name = db_path + "/estore_part_" + std::to_string(partition_id);
  auto ebe_name = backup_path + "/estore_bak_part_"
                              + std::to_string(partition_id);

  rocksdb::IOStatus rs;
  // Open the backup engine of vertex db.
  rs = rocksdb::BackupEngine::Open(backup_env,
        rocksdb::BackupEngineOptions(vbe_name),
        &vtx_be);
  LOG_RETURN_IF_STATUS_NOT_OK(rs);
  // Open the backup engine of edge db.
  rs = rocksdb::BackupEngine::Open(backup_env,
        rocksdb::BackupEngineOptions(ebe_name),
        &edge_be);
  LOG_RETURN_IF_STATUS_NOT_OK(rs);

  // Restore vertex db from backup with the specified backup id
  rs = vtx_be->RestoreDBFromBackup(bk_info.vertex_bid, vdb_name, vdb_name);
  LOG_RETURN_IF_STATUS_NOT_OK(rs)
  // Restore edge db from backup with the specified backup id
  rs = edge_be->RestoreDBFromBackup(bk_info.edge_bid, edb_name, edb_name);
  LOG_RETURN_IF_STATUS_NOT_OK(rs)

  rocksdb::Status s;
  // Open vertex db
  s = rocksdb::DB::Open(options, vdb_name, &vtx_db);
  LOG_RETURN_IF_STATUS_NOT_OK(s);
  // Open edge db
  s = rocksdb::DBWithTTL::Open(options, edb_name, &edge_db, ttl_in_seconds);
  LOG_RETURN_IF_STATUS_NOT_OK(s);

  is_open = true;
  return true;
}

bool SampleStore::PartitionedDB::Close() {
  rocksdb::Status s;
  s = vtx_db->Close();
  LOG_RETURN_IF_STATUS_NOT_OK(s);
  s = edge_db->Close();
  LOG_RETURN_IF_STATUS_NOT_OK(s);

  is_open = false;
  // TODO(@xmqin): cleanup backup engine.
  return true;
}

}  // namespace storage
}  // namespace dgs
