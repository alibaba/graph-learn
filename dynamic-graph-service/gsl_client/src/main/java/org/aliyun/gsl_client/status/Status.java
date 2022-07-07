package org.aliyun.gsl_client.status;

public class Status {

  private ErrorCode code;

  public Status(ErrorCode code) {
    this.code = code;
  }

  public boolean ok() {
    return code.ok();
  }

  public int getCode() {
    return code.getValue();
  }
}
