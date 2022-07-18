# Graph Operator

GraphLearn provides two Python interfaces, which we name "Graph Operator" and "GSL (Graph Sampling Language)".

Graph operator is the immediate execution of an arithmetic on a GraphLearn graph, specifying the input to the arithmetic and waiting for the output of the arithmetic.
GSL is a graph query language. A Query string together the input and output of multiple operators to form a directed acyclic data stream. An operator of a Query is derived from the traversal of a vertex or edge in the graph, and the Query can be executed repeatedly until the complete graph is traversed, producing a sequence of outputs.

- How do I choose a Python interface?

  If you need to traverse a graph, we recommend using GSL, which is backed by an efficient computational graph execution engine that helps you concurrently execute Query repeatedly. </br

  If using GraphLearn's model interface, use GSL.</br>
  If the input to the operator does not come from the graph, it is recommended to use the graph operator, and GSL's Query only supports traversal from the graph as input. </br>

This chapter describes the graph operator. To use GSL, please read the chapter on GSL. </br>

**Note: The graph operator must be called after `g.init()`.**</br>