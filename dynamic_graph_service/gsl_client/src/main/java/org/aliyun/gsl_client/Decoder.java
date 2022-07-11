package org.aliyun.gsl_client;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.aliyun.dgs.AttributeValueTypeRep;

// TODO(@Seventeen17): this is a temporary implementation
public class Decoder {
  private Map<Short, ArrayList<Integer>> featTypes = new HashMap<>();
  private Map<Short, ArrayList<Integer>> featDims = new HashMap<>();

  public Decoder() {}

  public void addFeatDesc(short key, ArrayList<String> types, ArrayList<Integer> dims) {
    ArrayList<Integer> t = new ArrayList<>();
    types.forEach(typeStr -> {
      switch (typeStr) {
        case "float": t.add(AttributeValueTypeRep.FLOAT32);
          break;
        case "string": t.add(AttributeValueTypeRep.STRING);
          break;
        case "long": t.add(AttributeValueTypeRep.INT64);
          break;
        default:
          break;
      }
    });
    featTypes.put(key, t);
    featDims.put(key, dims);
  }

  public ArrayList<Integer> getFeatDims(short key) {
    return featDims.get(key);
  }

  public ArrayList<Integer> getFeatTypes(short key) {
    return featTypes.get(key);
  }
}
