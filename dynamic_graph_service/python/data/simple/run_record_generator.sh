#!/usr/bin/env bash
HERE=$(cd "$(dirname "$0")";pwd)

if [ ! -d "$HERE/generated" ]; then
  mkdir -p $HERE/generated
fi


# Generate records contains both vertices and edges.
#     For each id in range [0, idcount), generate #degree vertices of vid=id with
#     different properties, and generate #degree of edges of src_vid=id,
#     dst_vid=id+degree with different properties. Total size is degree * idcount * 2.
#
#     Vtype and etype are fixed set as 0 and 1.
#     Schema for vertex and edge can be set in ReocrdGenerator with args \v_decoder and \e_decoder.
#     Timestamp is random distribution with low=min_timestamp and high=max_timestamp.
#     Zipf and power-low distribution will be supported later.
#
#     Vertex record example: 0(vtype)\t2(vid)\t10010(timestamp)\tattr_0:attr_1(attributes)
#     Edge record example: 1(etype)\t2(src_vid)\t3(dst_vid)\t10011(timestamp)\tattr_0:attr_1(attributes)
#       parameters:
#         --vtype: integer, vertex type
#         --etype: integer, edge type
#         --vbytes: string attribute bytes size of vertex
#         --ebytes: string attributes bytes size of edge
#         --idcount: id range.
#         --degree: for each id, the fanout of vertices and edges.
#         --min_timestamp: min timestamp.
#         --max_timestamp: max timestamp.
#         --shuffle_size: number of records to shuffle in the buffer,
#           when shuffle_size < total size(degree * idcount * 2). shuffle per batch with shuffle_size,
#           shuffle_size >= total size, shuffle the whole data,
#           shuffle_size == 0, without shuffle
#         --task_count: file partitions

python $HERE/record_generator.py --vtype=0 --etype=4 --vbytes=116 --ebytes=0 --idcount=10000000 --degree=10 --min_timestamp=1000 --max_timestamp=2000 --shuffle_size=10 --task_count=36
