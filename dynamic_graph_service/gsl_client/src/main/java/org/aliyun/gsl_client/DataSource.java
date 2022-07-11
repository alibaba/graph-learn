package org.aliyun.gsl_client;


public abstract class DataSource {

  public abstract Integer next();

  public abstract boolean hasNext();

}
