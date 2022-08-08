package org.aliyun.gsl_client;

import org.aliyun.dgs.EdgeRecordRep;
import org.aliyun.dgs.VertexRecordRep;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.status.ErrorCode;
import org.aliyun.dgs.EntryRep;
import org.aliyun.dgs.QueryResponseRep;
import org.aliyun.dgs.RecordBatchRep;
import org.aliyun.dgs.RecordRep;
import org.aliyun.dgs.RecordUnionRep;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Vector;

/**
 * Value contains the input of query and a byte buffer contains the
 * output of query executed once, which is represented as flatbuffer table
 * `QueryResponseRep`.
 */
public class Value {
  private Query query;
  private Long input;
  private ByteBuffer bv = null;
  private QueryResponseRep rep = null;

  /**
   * Constructor for Value with query input and query output as byte array.
   * @param query(Query), given query.
   * @param input(Long), query input
   * @param content(byte[]), query output, the byte array is data of flatbuffer
   * table `QueryResponseRep`.
   * @throws UserException
   */
  public Value(Query query, Long input, byte[] content) throws UserException {
    this.query = query;
    this.input = input;
    bv = ByteBuffer.wrap(content);
    try {
      rep = QueryResponseRep.getRootAsQueryResponseRep(bv);
    } catch (Exception e) {
      throw new UserException(ErrorCode.HTTP_ERROR, "Query Response is invalid.");
    }
  }

  /**
   * Constructor for Value with query input and query output as ByteBuffer.
   * @param query(Query), given query.
   * @param input(Long), query input.
   * @param buf(ByteBuffer), query output, the ByteBuffer is data of flatbuffer
   * table `QueryResponseRep`.
   */
  public Value(Query query, Long input, ByteBuffer buf) {
    this.query = query;
    this.input = input;
    rep = QueryResponseRep.getRootAsQueryResponseRep(buf);
  }

  /**
   * Clear the data in byte buffer.
   */
  public void clear() {
    bv.clear();
  }

  /**
   * Number of entries in QueryResponse. Each entry contains the sampling
   * results for one vertex with corresponding sampling op.
   * @return int, number of entities, each entry is value of flatbuffer table
   * `EntryRep`.
   */
  public int size() {
    return rep.resultsLength();
  }

  /**
   * Format the query results as EgoGraph.
   * @param query(Query), the executed query.
   * @return EgoGraph
   */
  public EgoGraph getEgoGraph() throws UserException{
    ArrayList<PlanNode> nodes = query.getPlan().getEgoGraphNodes();
    ArrayList<Integer> vtypes = new ArrayList<Integer>();
    ArrayList<Integer> vops = new ArrayList<Integer>();
    ArrayList<Integer> hops = new ArrayList<Integer>();
    ArrayList<Integer> eops = new ArrayList<Integer>();
    for (PlanNode node : nodes) {
      if (node.getKind().equals("EDGE_SAMPLER")) {
        eops.add(node.getId());
        hops.add(node.getParam("fanout"));
      } else if (node.getKind().equals("VERTEX_SAMPLER")){
        vtypes.add(node.getParam("vtype"));
        vops.add(node.getId());
      } else {
        eops.add(node.getId());
        hops.add(1);  // Fanout=1 for source node.
      }
    }
    EgoGraph egoGraph = new EgoGraph(vtypes, vops, hops, eops);
    return feedEgoGraph(egoGraph);
  }

  public EgoGraph getEgoGraph(ArrayList<Integer> vtypes,
                              ArrayList<Integer> vops,
                              ArrayList<Integer> hops,
                              ArrayList<Integer> eops) {
    EgoGraph egoGraph = new EgoGraph(vtypes, vops, hops, eops);
    return feedEgoGraph(egoGraph);
  }

  public EgoGraph feedEgoGraph(EgoGraph egoGraph) {
    egoGraph.addVids(0, input);
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

        egoGraph.addFeatures(entry.opid(), entry.vid(), vrec);
      }
    }
    return egoGraph;
  }
}
