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

#include "graphlearn/platform/local/local_file_system.h"

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>

#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/io/line_iterator.h"
#include "graphlearn/common/io/path_util.h"
#include "graphlearn/common/io/value.h"
#include "graphlearn/common/string/numeric.h"
#include "graphlearn/common/string/string_tool.h"

namespace graphlearn {

class LocalByteStreamAccessFile : public ByteStreamAccessFile {
public:
  LocalByteStreamAccessFile(const std::string& file_name,
                            uint64_t offset,
                            std::ifstream* f)
    : ByteStreamAccessFile(offset),
      file_name_(file_name), file_(f) {
    file_->seekg(offset_);
  }

  ~LocalByteStreamAccessFile() override {
    file_->close();
    delete file_;
  }

  Status Read(size_t n, LiteString* result, char* buffer) override {
    if (file_->good()) {
      file_->read(buffer, n);
    } else if (file_->eof()) {
      return error::OutOfRange("Read to end of the file");
    } else {
      return error::Internal("Read local file failed: " + file_name_);
    }

    if (file_->good() || file_->eof()) {
      uint64_t ret = file_->gcount();
      if (ret > 0) {
        *result = LiteString(buffer, ret);
        offset_ += ret;
        return Status::OK();
      } else {
        return error::OutOfRange("Read to end of the file");
      }
    } else {
      return error::Internal("Read local file failed: " + file_name_);
    }
  }

private:
  std::string    file_name_;
  std::ifstream* file_;
};

class LocalStructuredAccessFile : public StructuredAccessFile {
public:
  LocalStructuredAccessFile(const std::string& file_name,
                            uint64_t offset, std::ifstream* f)
    : StructuredAccessFile(offset),
      file_name_(file_name),
      line_(nullptr) {
    file_ = new LocalByteStreamAccessFile(file_name, 0, f);
    line_ = new io::LineIterator(file_, 2 * 1024 * 1024);

    Status s = Seek(offset_);
    if (!s.ok()) {
      LOG(ERROR) << "Invalid seek offset:" << offset;
    }
    s = ParseSchema(header_);
    if (!s.ok()) {
      LOG(ERROR) << "Invalid schema:" << header_;
    }
  }

  ~LocalStructuredAccessFile() override {
    delete file_;
    delete line_;
  }

  Status Read(io::Record* result) override {
    std::string tmp;
    Status s = line_->Next(&tmp);
    if (s.ok()) {
      ParseRecord(tmp, result);
    }
    return s;
  }

  const io::Schema& GetSchema() const override {
    return schema_;
  }

private:
  Status Seek(uint64_t offset) {
    std::string tmp;
    Status s = line_->Next(&tmp);
    if (s.ok()) {
      header_ = tmp;   // the first line is schema
    } else {
      return s;
    }

    uint64_t consumed = 0;
    while (consumed < offset && s.ok()) {
      ++consumed;
      s = line_->Next(&tmp);
    }
    return s;
  }

  Status ParseSchema(const std::string& header) {
    std::vector<std::string> items = strings::Split(header, '\t');
    for (const std::string& item : items) {
      std::vector<std::string> kv = strings::Split(item, ':');
      if (kv.size() != 2) {
        LOG(ERROR) << "Invalid schema:" << header;
        return error::InvalidArgument("Invalid schema.");
      }
      LiteString sp(kv[1]);
      strings::StripContext(&sp);
      schema_.Append(kv[0], io::ToDataType(sp.ToString()));
    }
    return Status::OK();
  }

  void ParseRecord(const std::string& line, io::Record* result) const {
    std::vector<std::string> items = strings::Split(line, '\t');
    if (items.size() != schema_.Size()) {
      return;
    }
    for (size_t i = 0; i < items.size(); ++i) {
      io::Value& v = (*result)[i];
      if (schema_.types[i] == DataType::kInt32) {
        strings::FastStringTo32(items[i].c_str(), &v.n.i);
      } else if (schema_.types[i] == DataType::kInt64) {
        strings::FastStringTo64(items[i].c_str(), &v.n.l);
      } else if (schema_.types[i] == DataType::kFloat ||
          schema_.types[i] == DataType::kDouble) {
        strings::FastStringToFloat(items[i].c_str(), &v.n.f);
      } else {
        v.s.Copy(items[i], true);
      }
    }
  }

private:
  std::string           file_name_;
  ByteStreamAccessFile* file_;
  io::LineIterator*     line_;
  std::string           header_;
  io::Schema            schema_;
};

class LocalWritableFile : public WritableFile {
public:
  LocalWritableFile(const std::string& file_name,
                    std::ofstream* f)
    : file_name_(file_name), file_(f) {
  }

  ~LocalWritableFile() override {
    delete file_;
  }

  Status Append(const LiteString& data) override {
    file_->write(data.data(), data.size());
    return Check();
  }

  Status Close() override {
    file_->close();
    return Check();
  }

  Status Flush() override {
    file_->flush();
    return Check();
  }

private:
  Status Check() {
    if (file_->good()) {
      return Status::OK();
    } else {
      return error::Internal("Write local file failed: " + file_name_);
    }
  }

private:
  std::string    file_name_;
  std::ofstream* file_;
};

Status LocalFileSystem::NewByteStreamAccessFile(
    const std::string& file_name, uint64_t offset,
    std::unique_ptr<ByteStreamAccessFile>* result) {
  std::string name = Translate(file_name);
  std::ifstream* f = new std::ifstream(name, std::ifstream::binary);
  if (f->good()) {
    result->reset(new LocalByteStreamAccessFile(name, offset, f));
    return Status::OK();
  } else {
    delete f;
    return error::InvalidArgument("Read local file failed");
  }
}

Status LocalFileSystem::NewStructuredAccessFile(
    const std::string& file_name, uint64_t offset, uint64_t end,
    std::unique_ptr<StructuredAccessFile>* result) {
  std::string name = Translate(file_name);
  std::ifstream* f = new std::ifstream(name, std::ifstream::binary);
  if (f->good()) {
    result->reset(new LocalStructuredAccessFile(name, offset, f));
    return Status::OK();
  } else {
    delete f;
    return error::InvalidArgument("Read local structured file failed");
  }
}

Status LocalFileSystem::NewWritableFile(
    const std::string& file_name,
    std::unique_ptr<WritableFile>* result) {
  std::string name = Translate(file_name);
  std::ofstream* f = new std::ofstream(name, std::ofstream::binary);
  if (f->good()) {
    result->reset(new LocalWritableFile(name, f));
    return Status::OK();
  } else {
    delete f;
    LOG(ERROR) << "Create local file failed: " << name;
    return error::InvalidArgument("Create local file failed");
  }
}

Status LocalFileSystem::ListDir(
    const std::string& path,
    std::vector<std::string>* result) {
  std::string trans_path = Translate(path);
  DIR* dir = ::opendir(trans_path.c_str());
  if (dir == nullptr) {
    return error::Internal(path + " open failed");
  }

  struct dirent* entry = ::readdir(dir);
  while (entry != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      // just ignore
    } else if (entry->d_type == DT_DIR) {
      result->push_back(name + "/");
    } else {
      result->push_back(name);
    }
    entry = ::readdir(dir);
  }
  ::closedir(dir);
  return Status::OK();
}

Status LocalFileSystem::GetFileSize(
    const std::string& file_name,
    uint64_t* size) {
  std::string name = Translate(file_name);
  struct stat st;
  if (stat(name.c_str(), &st) != 0) {
    *size = 0;
    return error::Internal("Get file size failed");
  } else {
    *size = st.st_size;
  }
  return Status::OK();
}

Status LocalFileSystem::GetRecordCount(
    const std::string& file_name,
    uint64_t* count) {
  std::vector<std::string> items = strings::Split(file_name, '#');
  if (items.size() >= 2) {
    int64_t c = 0;
    if (strings::FastStringTo64(items.back().c_str(), &c)) {
      *count = c;
      return Status::OK();
    }
  }

  std::ifstream in(file_name);
  if (!in) {
    return error::InvalidArgument("File not exist");
  }

  uint64_t num = 0;
  std::string line;
  while (std::getline(in, line)) {
    num++;
  }
  in.close();
  *count = num - 1;  // The first line is schema
  return Status::OK();
}

Status LocalFileSystem::FileExists(const std::string& file_name) {
  std::string name = Translate(file_name);
  if (access(name.c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return error::NotFound(file_name + " not found");
}

Status LocalFileSystem::DeleteFile(const std::string& file_name) {
  std::string name = Translate(file_name);
  if (unlink(name.c_str()) != 0) {
    LOG(ERROR) << "Delete local file failed: " << name;
    return error::Internal("Delete file failed");
  }
  return Status::OK();
}

Status LocalFileSystem::CreateDir(const std::string& dir_name) {
  std::string name = Translate(dir_name);
  if (access(name.c_str(), F_OK) == 0) {
    return error::AlreadyExists("Directory already exists");
  }

  if (mkdir(name.c_str(), 0755) != 0) {
    LOG(ERROR) << "Create local directory failed: " << name;
    return error::Internal("Create directory failed");
  }
  return Status::OK();
}

Status LocalFileSystem::DeleteDir(const std::string& dir_name) {
  std::string name = Translate(dir_name);
  if (rmdir(name.c_str()) != 0) {
    LOG(ERROR) << "Delete local directory failed: " << name;
    return error::Internal("Delete directory failed");
  }
  return Status::OK();
}

std::string LocalFileSystem::Translate(const std::string& path) const {
  return io::GetFilePath(path);
}

REGISTER_FILE_SYSTEM("", LocalFileSystem);

}  // namespace graphlearn
