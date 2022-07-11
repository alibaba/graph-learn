package org.aliyun.gsl_client.exception;

public class HttpException extends Exception {
  private int code;

  public HttpException(Exception e) {
    super(e);
  }

  public HttpException(int code, String message) {
    super(message);
    this.setCode(code);
  }

  public HttpException(int code, String message, Throwable cause) {
    super(message, cause);
    this.setCode(code);
  }

  public int getCode() {
    return code;
  }

  public void setCode(int code) {
    this.code = code;
  }
}
