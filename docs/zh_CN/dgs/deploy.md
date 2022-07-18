# 服务部署

> （完善中）

Dynamic-Graph-Service基于k8s进行部署.

整个服务包含以下子服务：

1. Kafka service
2. DataLoader
3. Sampling Workers
4. Serving Workers
5. Coordinator

6. Java Client
7. Tensorflow model service

1～5的部署参考[k8s cluster deployment](../../../dynamic_graph_service/k8s/charts/dgs/README.md)