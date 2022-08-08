package org.aliyun.gsl_client.predict;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;

import org.aliyun.dgs.VertexRecordRep;
import org.aliyun.dgs.AttributeRecordRep;
import org.aliyun.dgs.AttributeValueTypeRep;



public class EgoGraph {
  private ArrayList<Integer> vtypes;  // src_vtype, hop1_vtype, hop2_vtype...
  private ArrayList<Integer> vops;  // src_opid, hop1_opid, hop2_opid;
  private ArrayList<Integer> eops;  // input_opid, hop1_opid, hop2_opid;
  private ArrayList<Integer> hops;  // hop size;
  private ArrayList<ArrayList<Long>> vids;  // src_vid, hop0_vids, hop1_vids
  private HashMap<Integer, HashMap<Long, VertexRecordRep>> vfeats;  // opid->{vid->features}

  private HashMap<Short, Integer> attrdims;  // attrTypeId -> attrDim

  public EgoGraph(ArrayList<Integer> vtypes,
                  ArrayList<Integer> vops,
                  ArrayList<Integer> hops,
                  ArrayList<Integer> eops) {
    this.vtypes = vtypes;
    this.vops = vops;
    this.hops = hops;
    this.eops = eops;
    vids = new ArrayList<ArrayList<Long>>();
    for (int i = 0; i < hops.size(); ++i) {
      vids.add(new ArrayList<Long>(hops.get(i)));
    }
    vfeats = new HashMap<>(vtypes.size());
    attrdims = new HashMap<>();
  }

  public void addVids(int opid, long vid) {
    int hopId = eops.indexOf(opid);
    vids.get(hopId).add(vid);
  }

  public void addFeatures(Integer opid, long vid, VertexRecordRep attrs) {
    if (vfeats.containsKey(opid)) {
      HashMap<Long, VertexRecordRep> feats = vfeats.get(opid);
      feats.put(vid, attrs);
    } else {
      HashMap<Long, VertexRecordRep> feats = new HashMap<Long, VertexRecordRep>();
      feats.put(vid, attrs);
      vfeats.put(opid, feats);

      attrs.attributesLength();
      for ( int idx = 0; idx < attrs.attributesLength(); ++idx) {
        org.aliyun.dgs.AttributeRecordRep aRep = attrs.attributes(idx);
        short attrTypeId = aRep.attrType();
        if (!attrdims.containsKey(attrTypeId)) {
          int length = aRep.valueBytesLength();
          int dim = 1;
          switch (aRep.valueType()) {
            case AttributeValueTypeRep.INT32_LIST: dim = length / 4;
            break;
            case AttributeValueTypeRep.INT64_LIST: dim = length / 8;
            break;
            case AttributeValueTypeRep.FLOAT32_LIST: dim = length / 4;
            break;
            case AttributeValueTypeRep.FLOAT64_LIST: dim = length / 8;
            break;
          }
          attrdims.put(attrTypeId, dim);
        }
      }
    }
  }

  public int numHops() {
    return hops.size();
  }

  public int getVtype(int idx) {
    return vtypes.get(idx);
  }

  public int getVtypeFromOpId(int opid) {
    int idx = vops.indexOf(opid);
    return vtypes.get(idx);
  }

  public ArrayList<Long> getVids(int idx) {
    return vids.get(idx);
  }

  public ByteBuffer getVfeat(int idx, long vid, int featKey) {
    int opId = vops.get(idx);
    HashMap<Long, VertexRecordRep> feats = vfeats.get(opId);
    VertexRecordRep feat = feats.get(vid);
    AttributeRecordRep attrRecord = feat.attributesByKey((short)featKey);
    ByteBuffer bb = attrRecord.valueBytesAsByteBuffer();
    return bb;
  }

  public int getAttrDim(int attrType, int defaultDim) {
    if (attrdims.containsKey((short)attrType)) {
      return attrdims.get((short)attrType);
    }
    return defaultDim;
  }
}
