package org.aliyun.gsl_client;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.status.ErrorCode;
import org.apache.commons.lang3.ObjectUtils.Null;
import org.aliyun.dgs.AttributeRecordRep;
import org.aliyun.dgs.AttributeValueTypeRep;
import org.aliyun.dgs.EdgeRecordRep;
import org.aliyun.dgs.VertexRecordRep;
import org.aliyun.dgs.EntryRep;
import org.aliyun.dgs.QueryResponseRep;
import org.aliyun.dgs.RecordBatchRep;
import org.aliyun.dgs.RecordRep;
import org.aliyun.dgs.RecordUnionRep;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

public class Value {
  private ByteBuffer bv = null;
  private QueryResponseRep rep = null;

  public Value(byte[] content) throws UserException {
    bv = ByteBuffer.wrap(content);
    try {
      rep = QueryResponseRep.getRootAsQueryResponseRep(bv);
    } catch (Exception e) {
      throw new UserException(ErrorCode.HTTP_ERROR, "Query Response is invalid.");
    }
  }

  public Value(ByteBuffer buf) {
    rep = QueryResponseRep.getRootAsQueryResponseRep(buf);
  }

  public void clear() {
    bv.clear();
  }

  public boolean vaild() {
    try {
      read();
    } catch (UserException e) {
      return false;
    }
    return true;
  }

  public int size() {
    return rep.resultsLength();
  }

  public EgoGraph getEgoGraph(Plan plan, Decoder d) {
    EgoGraph egoGraph = new EgoGraph(plan);
    egoGraph.addVids(0, 0);
    for (int i = 0; i < size(); ++i) {
      EntryRep entry = rep.results(i);
      RecordBatchRep batch = entry.value();
      int batchSize = batch.recordsLength();
      RecordRep record = batch.records(0);
      byte recordType = record.recordType();
      if (recordType == RecordUnionRep.EdgeRecordRep) {
        for (int j = 0; j < batchSize; ++j) {
          RecordRep rec = batch.records(j);
          EdgeRecordRep erec = (EdgeRecordRep)rec.record(new EdgeRecordRep());
          egoGraph.addVids(entry.opid(), erec.dstId());
        }
      } else {
        RecordRep rec = batch.records(0);
        VertexRecordRep vrec = (VertexRecordRep)rec.record(new VertexRecordRep());

        ArrayList<ByteBuffer> feats = new ArrayList<>();
        int nFeats = d.getFeatTypes(egoGraph.getVtypeFromOpId(entry.opid())).size();
        for (int featIdx = 0; featIdx < nFeats; ++featIdx) {
          AttributeRecordRep attrs = vrec.attributesVector().get((short)featIdx);
          ByteBuffer bb = attrs.valueBytesAsByteBuffer();
          feats.add(bb);
        }
        egoGraph.addFeatures(entry.opid(), entry.vid(), feats);
      }
    }
    return egoGraph;
  }

  public void read() throws UserException {
    int size = rep.resultsLength();

    for (int i = 0; i < size; ++i) {
      EntryRep entry = rep.results(i);
      StringBuilder builder = new StringBuilder(512);
      builder.append("OpId: ");
      builder.append(entry.opid());
      builder.append("\t");
      builder.append("Vid: ");
      builder.append(entry.vid());
      builder.append("\n");

      RecordBatchRep batch = entry.value();
      int batchSize = batch.recordsLength();
      builder.append("----");
      builder.append("batch size: ");
      builder.append(batchSize);
      builder.append("\n");

      byte recordType = 0;
      if (batchSize > 0) {
        RecordRep record = batch.records(0);
        recordType = record.recordType();
      } else {
        throw new UserException(ErrorCode.HTTP_ERROR, "RecordBatch is empty.");
      }
      if (recordType == 0) {
        throw new UserException(ErrorCode.HTTP_ERROR, "recordType is unkown.");
      }
      if (recordType == 2) {
        for (int j = 0; j < batchSize; ++j) {
          RecordRep rec = batch.records(j);
          builder.append("----");
          EdgeRecordRep erec = (EdgeRecordRep)rec.record(new EdgeRecordRep());
          builder.append(erec.srcId());
          builder.append("\t");
          builder.append(erec.dstId());
          builder.append("\n");
        }
      } else {
        for (int j = 0; j < batchSize; ++j) {
          RecordRep rec = batch.records(j);
          builder.append("----");
          VertexRecordRep vrec = (VertexRecordRep)rec.record(new VertexRecordRep());
          builder.append(vrec.vid());
          builder.append("\n");
        }
      }
      System.out.println(builder);
    }
  }
}
