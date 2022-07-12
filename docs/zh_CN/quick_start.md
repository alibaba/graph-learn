# 快速开始
Graph-Learn 包含两大部分, **GraphLearn-Training** 和 **Dynamic-Graph-Service**.

## GraphLearn-Training

源码位置：

```
git clone https://github.com/alibaba/graph-learn.git
cd graphlearn
```

GraphLearn-Training运行GNN模型训练任务，训练完的模型可以：
1. 保存Embedding，进行向量检索
2. 保存Checkpoint，并部署到TF Model Service上，做在线预测。


以下是一个单机GraphLearn-Training模型示例。

1. 拉取docker镜像

```
docker run -ti --name gl_docker --net=host graphlearn/graphlearn bash
```

2. 更新wheel

```
pip install -U graph-learn
```

> 从源码构建wheel包 [ref](gl/install.md)

3. train the model

```
cd examples
# Prepare data
cd data && python cora.py
cd ../tf/ego_sage && python train_supervised.py
```

使用kubeflow运行分布式GraphLearn-Training任务的示例：

1. 配置yaml

```
cd k8s
vim dist.yaml
# config ${your_job_name}, ${your_namespace},  ${your_volume} and ${your_pvc_name}
```

2. 提交任务

```
kubectl apply -f dist.yaml
```

> 除了训练GNN model, GraphLearn-Training提供了多层的用户接口，方便用户描述图、采样和模型开发，参考[ref](gl/quick_start.md).

## Dynamic-Graph-Service

源码位置：

```
git clone https://github.com/alibaba/graph-learn.git
cd dynamic_graph_service
```

动态图采样服务依赖于GraphLearn-Training训练出来的模型，我们提供了一个训练+在线采样+在线预测的端到端的示例，参考[ref](tutorial.md)