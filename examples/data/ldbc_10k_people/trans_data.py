import os
import numpy as np
import sys


def preprocess(dataset):
    node_file = "{}/person_degree.csv".format(dataset)
    lines = np.genfromtxt("{}/node_table_degree".format(dataset), delimiter="\t",
                  dtype=np.dtype(str))
    with open(node_file, "w") as f:
        header="id"
        for i in range(100):
            header = header + "|attr-" + str(i)
        f.write(header + "\n")
        for line in lines:
            new_line = line[0]
            attrs = line[1].split(":")
            for a in attrs:
                new_line = new_line + "|" + a
            f.write(new_line + "\n")


if __name__ == "__main__":
    prefix = sys.argv[1]
    preprocess(prefix)