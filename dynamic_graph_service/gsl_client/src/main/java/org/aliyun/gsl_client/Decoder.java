package org.aliyun.gsl_client;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.aliyun.dgs.AttributeValueTypeRep;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.status.ErrorCode;

/**
 * Decoder is used for descirbing the features data-types and dimensions
 * of each type of Vertex and Edge.
 */
public class Decoder {
  private Schema schema = null;
  // key: vertex type or edge type, value: list of `AttributeValueTypeRep`s
  private Map<Integer, ArrayList<Integer>> featTypes = new HashMap<>();
  // key: vertex type or edge type, value: list of attribute dimensions
  private Map<Integer, ArrayList<Integer>> featDims = new HashMap<>();

  public Decoder() {}

  public Decoder(Graph g) throws UserException {
    this.schema = g.getSchema();
  }

  /**
   * Add feature types and dims for Vertex or Edge.
   * @param key(String), Vertex type or Edge type.
   * @param types(ArrayList<String>), list of feature types, each element
   * is one of "float", "string" or "long".
   * @param dims(ArrayList<Integer>), list of feature dims for corresponding
   * position of feature tyep.
   * For example, feature type "float" with dim 128 represent a pretrained
   * embedding vector with 128 floats.
   */
  public void addFeatDesc(String key,
                          ArrayList<String> types,
                          ArrayList<Integer> dims) throws UserException {
    int keyId = schema.getTypeDef(key).getTypeId();
    addFeatDesc(keyId, types, dims);
  }

  public void addFeatDesc(int keyId,
                          ArrayList<String> types,
                          ArrayList<Integer> dims) throws UserException {
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
    if (t.size() != types.size()) {
      throw new UserException(ErrorCode.UNDEFINED_ERROR,
            "Only \"string\", \"flaot\" and \"long\" feature types are supported.");
    }
    if (t.size() != dims.size()) {
      throw new UserException(ErrorCode.PAESE_ERROR,
        "Must assign types and dims with same size.");
    }
    featTypes.put(keyId, t);
    featDims.put(keyId, dims);
  }

  /**
   * Get feature dims for given Vertex or Edge type.
   * @param key(int), Vertex or Edge encoded type.
   * @return ArrayList<Integer>, feature dims.
   */
  public ArrayList<Integer> getFeatDims(int key) {
    return featDims.get(key);
  }

  /**
   * Get feature types for given Vertex or Edge type.
   * @param key(int), Vertex or Edge encoded type.
   * @return ArrayList<Short>, feature types as AttributeValeuTypeRep.
   */
  public ArrayList<Integer> getFeatTypes(int key) {
    return featTypes.get(key);
  }
}
