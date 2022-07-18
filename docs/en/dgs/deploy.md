# Service Deployment

TODO(@Seventeen17): Complete me

Dynamic-Graph-Service is deployed based on k8s.

The whole service contains the following sub-services.

1. Kafka service
2. DataLoader
3. Sampling Workers
4. Serving Workers
5. Coordinator

6. Java Client
7. Tensorflow model service

Deply the 1~5 services, ref to [k8s cluster deployment](../../../dynamic_graph_service/k8s/charts/dgs/README.md)
