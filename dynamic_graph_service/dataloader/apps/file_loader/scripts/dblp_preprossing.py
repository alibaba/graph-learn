# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import os

delimiter = '&'
attr_delimiter = ";"

output_dir = "./"
pattern_file = "dblp.pattern"
data_file = "dblp.data"

year_min = 1949
year_max = 2019


def write_pattern_file():
  with open(os.path.join(output_dir, pattern_file), 'w') as fw:
    fw.write("#VERTEX:paper{d}#ID{d}timestamp{d}attrs\n".format(d=delimiter))
    fw.write("#VERTEX:author{d}#ID{d}timestamp\n".format(d=delimiter))
    fw.write("#VERTEX:venue{d}#ID{d}timestamp\n".format(d=delimiter))
    fw.write("#EDGE:published{d}#SRC:paper{d}#DST:venue{d}timestamp\n".format(d=delimiter))
    fw.write("#EDGE:written{d}#SRC:paper{d}#DST:author{d}timestamp\n".format(d=delimiter))


def process_dataset(dataset):
  paper_set = set()
  author_set = set()
  venue_set = set()
  year_set = set()
  sort_buf = []

  with open(dataset, 'r') as fr:
    line_reader = csv.reader(fr, delimiter=',', quotechar='"')
    next(line_reader)
    i = 0
    for line in line_reader:
      i += 1
      author, title, year, volume, venue, number, _, _ = line
      # Check if title contains delimiter.
      if (title.find(delimiter)) != -1:
        print("WARNING: Title has {delim} ! -> {title}".format(delim=delimiter, title=title))
      # Check if title contains attribute delimiter.
      if (title.find(attr_delimiter)) != -1:
        print("WARNING: Title has {attr_delim} ! -> {title}".format(attr_delim=attr_delimiter, title=title))
      if author == '' or title == '' or int(year) > year_max or int(year) < year_min or venue == '':
        continue
      paper_set.add(title)
      author_set.add(author)
      venue_set.add(venue)
      year_set.add(int(year))
      sort_buf.append((author, title, int(year), volume, venue, number))
  print("#author:%d #paper:%d #venue:%d" % (len(author_set), len(paper_set), len(venue_set)))

  a2i = {a: i for i, a in enumerate(author_set)}
  v2i = {v: (i + len(author_set)) for i, v in enumerate(venue_set)}
  p2i = {p: (i + len(author_set) + len(venue_set)) for i, p in enumerate(paper_set)}
  sort_buf = sorted(sort_buf, key=lambda x: x[2])

  with open(os.path.join(output_dir, data_file), 'w') as fw:
    for a in author_set:
      fw.write("author{d}{author_id}{d}0\n".format(d=delimiter, author_id=a2i[a]))
    for v in venue_set:
      fw.write("venue{d}{venue_id}{d}0\n".format(d=delimiter, venue_id=v2i[v]))
    for line in sort_buf:
      author, title, year, volume, venue, number = line
      fw.write("paper{d}{paper_id}{d}{year}{d}{title}{ad}{volume}{ad}{number}\n"
               .format(d=delimiter, ad=attr_delimiter, paper_id=p2i[title], year=year,
                       title=title, volume=volume, number=number))
      fw.write("published{d}{paper_id}{d}{venue_id}{d}{year}\n"
               .format(d=delimiter, paper_id=p2i[title], venue_id=v2i[venue], year=year))
      fw.write("written{d}{paper_id}{d}{author_id}{d}{year}\n"
               .format(d=delimiter, paper_id=p2i[title], author_id=a2i[author], year=year))

  print("Generated processed files of dataset: {}.".format(dataset))
  print("-> pattern file: \"{}\"".format(os.path.join(output_dir, pattern_file)))
  print("-> data file: \"{}\"".format(os.path.join(output_dir, data_file)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='DBLP Dataset Preprocessing Tool')
  parser.add_argument('--dataset', action="store", dest="dataset",
                      help="The csv file path of dblp dataset.")
  parser.add_argument('--start-year', action="store", dest="start_year",
                      help="only records after the start year will be processed.")
  parser.add_argument('--end-year', action="store", dest="end_year",
                      help="only records before the end year will be processed.")
  parser.add_argument('--output-dir', action="store", dest="output_dir",
                      help="The output directory of processed files.")
  args = parser.parse_args()

  if args.dataset is None:
    raise RuntimeError("The dblp dataset file must be specified!")
  if not os.path.exists(args.dataset):
    raise RuntimeError("Missing dataset file: {}!".format(args.dataset))

  if args.start_year is not None:
    year_min = int(args.start_year)
  if args.end_year is not None:
    year_max = int(args.end_year)

  if args.output_dir is not None:
    output_dir = args.output_dir

  write_pattern_file()
  process_dataset(args.dataset)
