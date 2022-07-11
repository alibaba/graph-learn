package org.aliyun.gsl_client.parser.schema;

import java.util.Objects;

public class RawEdgeRelation {
    protected String edgeName;
    protected String srcName;
    protected String dstName;

    public RawEdgeRelation(String edgeName, String srcName, String dstName) {
        this.edgeName = edgeName;
        this.srcName = srcName;
        this.dstName = dstName;
    }

    public String getEdgeName() {
        return this.edgeName;
    }

    public String getSrcName() {
        return this.srcName;
    }

    public String getDstName() {
        return this.dstName;
    }

    @Override
    public String toString() {
        return this.srcName + "->" + this.edgeName + "->" + this.dstName;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        RawEdgeRelation rawEdgeRelation = (RawEdgeRelation) o;

        if (!Objects.equals(this.edgeName, rawEdgeRelation.edgeName)) {
            return false;
        }
        if (!Objects.equals(this.srcName, rawEdgeRelation.srcName)) {
            return false;
        }
        return Objects.equals(this.srcName, rawEdgeRelation.dstName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.edgeName, this.srcName, this.dstName);
    }
}
