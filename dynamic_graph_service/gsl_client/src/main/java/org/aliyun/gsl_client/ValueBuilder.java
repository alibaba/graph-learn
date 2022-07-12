package org.aliyun.gsl_client;

import org.aliyun.graphlearn.AttributeRecordRep;
import org.aliyun.graphlearn.AttributeValueTypeRep;
import org.aliyun.graphlearn.EdgeRecordRep;
import org.aliyun.graphlearn.VertexRecordRep;

import com.google.flatbuffers.FlatBufferBuilder;

import org.aliyun.graphlearn.EntryRep;
import org.aliyun.graphlearn.QueryResponseRep;
import org.aliyun.graphlearn.RecordBatchRep;
import org.aliyun.graphlearn.RecordRep;
import org.aliyun.graphlearn.RecordUnionRep;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This is a helper class for created `Value` manually instead of quering from
 * graph service.
 */
public class ValueBuilder {
  private FlatBufferBuilder builder = new FlatBufferBuilder(0);
  private ByteBuffer buf;
  private int[] results;
  private int cursor;
  private Query query;
  private long input;
  private Decoder decoder;

  public ValueBuilder(Query query, int size, Decoder decoder, long input) {
    this.query = query;
    results = new int[size];
    cursor = 0;
    this.decoder = decoder;
    this.input = input;
  }

  public Value finish() {
    int val = QueryResponseRep.createResultsVector(builder, results);

    QueryResponseRep.startQueryResponseRep(builder);
    QueryResponseRep.addResults(builder, val);
    int orc = QueryResponseRep.endQueryResponseRep(builder);
    builder.finish(orc);

    buf = builder.dataBuffer();
    return new Value(query, input, buf);
  }

  private int addFloatAttributes(int dim, short idx) {
    float[] floatAttrs = new float[dim];
    Arrays.fill(floatAttrs, 1F);
    ByteBuffer bb = ByteBuffer.allocate(4 * dim);
    FloatBuffer fbb= bb.asFloatBuffer();
    fbb.put(floatAttrs);
    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);
    AttributeRecordRep.addAttrType(builder, idx);  // assume we use incremental encode of attr type
    AttributeRecordRep.addValueType(builder, AttributeValueTypeRep.FLOAT32);  // FLOAT32=6 in AttributeValueTypeRep
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    // builder.finish(attr);
    return attr;
  }

  private int addLongAttributes(int dim, short idx) {
    long[] longAttrs = new long[dim];
    Arrays.fill(longAttrs, 1L);
    ByteBuffer bb = ByteBuffer.allocate(8 * dim);
    LongBuffer lbb= bb.asLongBuffer();
    lbb.put(longAttrs);
    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);
    AttributeRecordRep.addAttrType(builder, idx);  // assume we use incremental encode of attr type
    AttributeRecordRep.addValueType(builder, AttributeValueTypeRep.INT64);  // INT64=5 in AttributeValueTypeRep
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    return attr;
  }

  private int addStringAttributes(int length, short idx) {
    char[] charArray = new char[length];
    Arrays.fill(charArray, ' ');
    String attrs = new String(charArray);
    ByteBuffer bb = ByteBuffer.wrap(attrs.getBytes());
    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);
    AttributeRecordRep.addAttrType(builder, idx);  // assume we use incremental encode of attr type
    AttributeRecordRep.addValueType(builder, AttributeValueTypeRep.STRING);  // =5 in AttributeValueTypeRep
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    return attr;
  }

  public int addVertex(short vtype, long vid) {
    ArrayList<Integer> featTypes = decoder.getFeatTypes(vtype);
    int[] offsets = new int[featTypes.size()];
    for (short idx = 0; idx < featTypes.size(); ++idx) {
      int dim = decoder.getFeatDims(vtype).get(idx);
      switch (featTypes.get(idx)) {
        case AttributeValueTypeRep.FLOAT32: offsets[idx] = addFloatAttributes(dim, idx);
                                            break;
        case AttributeValueTypeRep.INT32: offsets[idx] = addLongAttributes(dim, idx);
                                            break;
        case AttributeValueTypeRep.STRING: offsets[idx] = addStringAttributes(dim, idx);
                                            break;
      }
    }
    int attr = VertexRecordRep.createAttributesVector(builder, offsets);

    VertexRecordRep.startVertexRecordRep(builder);
    VertexRecordRep.addVtype(builder, vtype);
    VertexRecordRep.addVid(builder, vid);
    VertexRecordRep.addAttributes(builder, attr);
    int v = VertexRecordRep.endVertexRecordRep(builder);

    RecordRep.startRecordRep(builder);
    RecordRep.addRecordType(builder, RecordUnionRep.VertexRecordRep);
    RecordRep.addRecord(builder, v);
    return RecordRep.endRecordRep(builder);
  }

  public int addEdge(short etype, short srcVtype, short dstVtype,
                     long srcId, long dstId) {
    int[] offsets = new int[1];
    offsets[0] = addFloatAttributes(1, (short)0);  // weight
    int attr = EdgeRecordRep.createAttributesVector(builder, offsets);

    EdgeRecordRep.startEdgeRecordRep(builder);
    EdgeRecordRep.addEtype(builder, etype);
    EdgeRecordRep.addSrcVtype(builder, srcVtype);
    EdgeRecordRep.addDstVtype(builder, dstVtype);
    EdgeRecordRep.addSrcId(builder, srcId);
    EdgeRecordRep.addDstId(builder, dstId);
    EdgeRecordRep.addAttributes(builder, attr);
    int e = EdgeRecordRep.endEdgeRecordRep(builder);
    RecordRep.startRecordRep(builder);
    RecordRep.addRecordType(builder, RecordUnionRep.EdgeRecordRep);
    RecordRep.addRecord(builder, e);
    return RecordRep.endRecordRep(builder);
  }

  public int addVertexRecords(int batchSize, short vtype, long vid) {
    int[] offs = new int[batchSize];
    for (int i = 0; i < batchSize; ++i) {
      offs[i] = addVertex(vtype, vid);
    }
    int val = RecordBatchRep.createRecordsVector(builder, offs);

    RecordBatchRep.startRecordBatchRep(builder);
    RecordBatchRep.addRecords(builder, val);
    RecordBatchRep.addPartition(builder, 2L);
    int orc = RecordBatchRep.endRecordBatchRep(builder);
    return orc;
  }

  public int addEdgeRecords(int batchSize,
                            short etype, short srcVtype, short dstVtype,
                            long vid) {
    int[] offs = new int[batchSize];
    for (int i = 0; i < batchSize; ++i) {
      offs[i] = addEdge(etype, srcVtype, dstVtype,
                        vid, vid * batchSize + i);
    }
    int val = RecordBatchRep.createRecordsVector(builder, offs);

    RecordBatchRep.startRecordBatchRep(builder);
    RecordBatchRep.addRecords(builder, val);
    RecordBatchRep.addPartition(builder, 2L);
    int orc = RecordBatchRep.endRecordBatchRep(builder);
    return orc;
  }

  public int addVopRes(int opid, short vtype, long vid, int batchSize) {
    int vec = addVertexRecords(batchSize, vtype, vid);

    EntryRep.startEntryRep(builder);
    EntryRep.addOpid(builder, opid);
    EntryRep.addVid(builder, vid);
    EntryRep.addValue(builder, vec);
    int orc = EntryRep.endEntryRep(builder);
    results[cursor] = orc;
    cursor += 1;
    return orc;
  }

  public int addEopRes(int opid, short etype, short srcType, short DstType,
                       long vid, int batchSize) {
    int vec = addEdgeRecords(batchSize, etype, srcType, DstType, vid);
    EntryRep.startEntryRep(builder);
    EntryRep.addOpid(builder, opid);
    EntryRep.addVid(builder, vid);
    EntryRep.addValue(builder, vec);
    int orc = EntryRep.endEntryRep(builder);
    results[cursor] = orc;
    cursor += 1;
    return orc;
  }
}
