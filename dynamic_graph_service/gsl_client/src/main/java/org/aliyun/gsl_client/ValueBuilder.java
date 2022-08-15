package org.aliyun.gsl_client;

import org.aliyun.dgs.AttributeRecordRep;
import org.aliyun.dgs.AttributeValueTypeRep;
import org.aliyun.dgs.EdgeRecordRep;
import org.aliyun.dgs.VertexRecordRep;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.AttrDef;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.parser.schema.TypeDef;

import com.google.flatbuffers.FlatBufferBuilder;

import org.aliyun.dgs.EntryRep;
import org.aliyun.dgs.QueryResponseRep;
import org.aliyun.dgs.RecordBatchRep;
import org.aliyun.dgs.RecordRep;
import org.aliyun.dgs.RecordUnionRep;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.List;


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
  private Schema schema;
  private int dim;

  public ValueBuilder(Query query, int size, Schema schema, long input, int dim) {
    this.query = query;
    results = new int[size];
    cursor = 0;
    this.schema = schema;
    this.input = input;
    this.dim = dim;
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

  private int addFloatAttributes(int dim, int attrTypeId) {
    float[] floatAttrs = new float[dim];
    Arrays.fill(floatAttrs, 1F);
    ByteBuffer bb = ByteBuffer.allocate(4 * dim);
    FloatBuffer fbb= bb.asFloatBuffer();
    fbb.put(floatAttrs);

    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);

    AttributeRecordRep.addAttrType(builder, (short) attrTypeId);
    int attrDataType = AttributeValueTypeRep.FLOAT32_LIST;
    if (dim == 1) {
      attrDataType = AttributeValueTypeRep.FLOAT32;
    }
    AttributeRecordRep.addValueType(builder, attrDataType);
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    return attr;
  }

  private int addLongAttributes(int dim, int attrTypeId) {
    long[] longAttrs = new long[dim];
    Arrays.fill(longAttrs, 1L);
    ByteBuffer bb = ByteBuffer.allocate(8 * dim);
    LongBuffer lbb= bb.asLongBuffer();
    lbb.put(longAttrs);
    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);
    AttributeRecordRep.addAttrType(builder, (short)attrTypeId);
    int attrDataType = AttributeValueTypeRep.INT64_LIST;
    if (dim == 1) {
      attrDataType = AttributeValueTypeRep.INT64;
    }
    AttributeRecordRep.addValueType(builder, attrDataType);
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    return attr;
  }

  private int addStringAttributes(int length, int attrTypeId) {
    char[] charArray = new char[length];
    Arrays.fill(charArray, ' ');
    String attrs = new String(charArray);
    ByteBuffer bb = ByteBuffer.wrap(attrs.getBytes());
    int val = AttributeRecordRep.createValueBytesVector(builder, bb);

    AttributeRecordRep.startAttributeRecordRep(builder);
    AttributeRecordRep.addAttrType(builder, (short)attrTypeId);
    AttributeRecordRep.addValueType(builder, AttributeValueTypeRep.STRING);
    AttributeRecordRep.addValueBytes(builder, val);
    int attr = AttributeRecordRep.endAttributeRecordRep(builder);
    return attr;
  }

  public int addVertex(short vtype, long vid) throws UserException {
    TypeDef typeDef = schema.getTypeDef(vtype);
    List<AttrDef> attrDefs = typeDef.getAttributes();

    int[] offsets = new int[attrDefs.size()];
    for (short idx = 0; idx < attrDefs.size(); ++idx) {

      switch (attrDefs.get(idx).getDataType()) {
        case FLOAT32_LIST: offsets[idx] = addFloatAttributes(dim, attrDefs.get(idx).getTypeId());
        break;
        case STRING: offsets[idx] = addStringAttributes(dim, attrDefs.get(idx).getTypeId());
        break;
        case INT64: offsets[idx] = addLongAttributes(1, attrDefs.get(idx).getTypeId());
        break;
        default:
        System.out.println("Add not existed attr type=" + attrDefs.get(idx).getDataType());
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
                     long srcId, long dstId) throws UserException {
    TypeDef typeDef = schema.getTypeDef(etype);
    List<AttrDef> attrDefs = typeDef.getAttributes();

    int[] offsets = new int[attrDefs.size()];
    for (short idx = 0; idx < attrDefs.size(); ++idx) {

      switch (attrDefs.get(idx).getDataType()) {
        case FLOAT32_LIST: offsets[idx] = addFloatAttributes(dim, attrDefs.get(idx).getTypeId());
        break;
        case STRING: offsets[idx] = addStringAttributes(dim, attrDefs.get(idx).getTypeId());
        break;
        case INT64: offsets[idx] = addLongAttributes(1, attrDefs.get(idx).getTypeId());
        break;
        default:
        break;
      }
    }
    int attr = VertexRecordRep.createAttributesVector(builder, offsets);

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

  public int addVertexRecords(int batchSize, short vtype, long vid) throws UserException {
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
                            long vid) throws UserException {
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

  public int addVopRes(int opid, short vtype, long vid, int batchSize) throws UserException {
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
                       long vid, int batchSize) throws UserException {
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
