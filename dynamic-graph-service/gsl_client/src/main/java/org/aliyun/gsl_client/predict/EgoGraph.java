package org.aliyun.gsl_client.predict;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.aliyun.gsl_client.parser.Plan;

public class EgoGraph {
  private ArrayList<Short> vtypes;  // src_vtype, hop1_vtype, hop2_vtype...
  private ArrayList<Integer> vops;  // src_opid, hop1_opid, hop2_opid;
  private ArrayList<Integer> eops;  // input_opid, hop1_opid, hop2_opid;
  private ArrayList<Integer> hops;  // hop size;
  private ArrayList<ArrayList<Long>> vids;  // src_vid, hop0_vids, hop1_vids
  private HashMap<Integer, HashMap<Long, ArrayList<ByteBuffer>>> vfeats;  // opid->{vid->features}
  // Split ByteBuffer

  public EgoGraph(Plan plan) {
    // TODO(@Seventeen17): parse from plan.
    vtypes = new ArrayList<Short>(Arrays.asList((short)0, (short)1, (short)1));
    vops = new ArrayList<Integer>(Arrays.asList(2, 4, 4));
    hops = new ArrayList<Integer>(Arrays.asList(1, 10, 5));
    eops = new ArrayList<Integer>(Arrays.asList(0, 1, 3));
    vids = new ArrayList<ArrayList<Long>>();
    for (int i = 0; i < hops.size(); ++i) {
      vids.add(new ArrayList<Long>(hops.get(i)));
    }
    vfeats = new HashMap<>(2);
  }

  public void addVids(int opid, long vid) {
    int hopId = eops.indexOf(opid);
    vids.get(hopId).add(vid);
  }

  public void addFeatures(Integer opid, long vid, ArrayList<ByteBuffer> attrs) {
    if (vfeats.containsKey(opid)) {
      HashMap<Long, ArrayList<ByteBuffer>> feats = vfeats.get(opid);
      feats.put(vid, attrs);
    } else {
      HashMap<Long, ArrayList<ByteBuffer>> feats = new HashMap<Long, ArrayList<ByteBuffer>>();
      feats.put(vid, attrs);
      vfeats.put(opid, feats);
    }
  }

  public int numHops() {
    return hops.size();
  }

  public short getVtype(int idx) {
    return vtypes.get(idx);
  }

  public short getVtypeFromOpId(int opid) {
    int idx = vops.indexOf(opid);
    return vtypes.get(idx);
  }

  public ArrayList<Long> getVids(int idx) {
    return vids.get(idx);
  }

  public ArrayList<ByteBuffer> getVfeats(int idx, Long vid) {
    int opId = vops.get(idx);
    HashMap<Long, ArrayList<ByteBuffer>> feats = vfeats.get(opId);
    ArrayList<ByteBuffer> feat = feats.get(vid);
    return feat;
  }
}
