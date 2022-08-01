# Service Deployment

The Dynamic-Graph-Service can be deployed on a [Kubernetes](https://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure

## Deploy kafka Queue Service
The Dynamic-Graph-Service uses [Kafka](https://kafka.apache.org/) queue service to store streaming graph updates and sampled results.
Before deploy a dgs service, you must deploy a kafka cluster first and create the following kafka queues:
- `dl2spl`: receiving graph updates from your data-source loaders, and consumed by sampling workers.
- `spl2srv`: receiving sampled results from sampling workers, and consumed by serving workers.

## Installing the Chart

Get repo info:
```shell
helm repo add dgs https://graphlearn.oss-cn-hangzhou.aliyuncs.com/charts/dgs/
helm repo update
```

Install the chart with release name `my-release`:

```shell
helm install my-release dgs/dgs \
    --set-file graphSchema=/path/to/schema/json/file \
    --set kafka.dl2spl.brokers=[your kafka broker list of dl2spl] \
    --set kafka.dl2spl.topic="your_kafka_topic_of_dl2spl" \
    --set kafka.dl2spl.partitions=your_kafka_partitions_of_dl2spl \
    --set kafka.spl2srv.brokers=[your kafka broker list of spl2srv] \
    --set kafka.spl2srv.topic="your_kafka_topic_of_spl2srv" \
    --set kafka.spl2srv.partitions=your_kafka_partitions_of_spl2srv
```

The graph schema must be specified from a json string or file by parameter `graphSchema`.
A [template](https://github.com/alibaba/graph-learn/blob/master/dynamic_graph_service/conf/schema.template.json)
schema file can be followed to write your customized graph schema.

The info of `dl2spl` and `spl2srv`  of your pre-deployed kafka cluster should be configured when you install the chart,
refer to [Kafka service parameters](#kafka-service-parameters)

These commands deploy Dynamic-Graph-Service on the Kubernetes cluster in the default configuration.
The [Parameters](#parameters) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Uninstalling the Chart

To uninstall/delete the `my-release` deployment:

```shell
helm delete my-release
```

The command removes all the Kubernetes components associated with the chart and deletes the release.

## Parameters

### Global parameters

| Name                       | Description                                                                             | Value                 |
| -------------------------- | --------------------------------------------------------------------------------------- | --------------------- |
| `kubeVersion`              | Override Kubernetes version                                                             | `""`                  |
| `nameOverride`             | String to partially override common.names.fullname                                      | `""`                  |
| `fullnameOverride`         | String to fully override common.names.fullname                                          | `""`                  |
| `clusterDomain`            | Default Kubernetes cluster domain                                                       | `cluster.local`       |
| `commonLabels`             | Labels to add to all deployed objects                                                   | `{}`                  |
| `commonAnnotations`        | Annotations to add to all deployed objects                                              | `{}`                  |
| `graphSchema`              | The json string of graph schema, **must** be set during installation                    | `""`                  |
| `configPath`               | The service configmap mount path                                                        | `"/dgs_conf"`         |
| `glog.toConsole`           | Specify whether program logs are written to standard error as well as to files          | `false`               |

### Kafka service parameters

| Name                        | Description                                                                   | Value                |
| --------------------------- | ----------------------------------------------------------------------------- | -------------------- |
| `kafka.dl2spl.brokers`      | Kafka brokers of processed graph updates from dataloader to sampling workers  | `["localhost:9092"]` |
| `kafka.dl2spl.topic`        | Kafka topic of (dataloader -> sampling workers)                               | `"record-batches"`   |
| `kafka.dl2spl.partitions`   | Topic partition number of (dataloader -> sampling workers)                    | `4`                  |
| `kafka.spl2srv.brokers`     | Kafka brokers of sampled updates from sampling workers to serving workers     | `["localhost:9092"]` |
| `kafka.spl2srv.topic`       | Kafka topic of (sampling workers -> serving workers)                          | `"sample-batches"`   |
| `kafka.spl2srv.partitions`  | Topic partition number of (sampling workers -> serving workers)               | `4`                  |

### Image parameters

| Name                        | Description                                                            | Value                |
| --------------------------- | ---------------------------------------------------------------------- | -------------------- |
| `image.registry`            | Core service image registry                                            | `"graphlearn"`       |
| `image.repository`          | Core service image repository                                          | `"dgs-core"`         |
| `image.tag`                 | Core service image tag (immutable tags are recommended)                | `"1.0.0"`            |
| `image.pullPolicy`          | Core service image pull policy                                         | `IfNotPresent`       |
| `image.pullSecrets`         | Specify docker-registry secret names as an array                       | `[]`                 |

### FrontEnd parameters

| Name                         | Description                                                              | Value                            |
| ---------------------------- | ------------------------------------------------------------------------ | -------------------------------- |
| `frontend.ingressHostName`   | The host name of external ingress                                        | `"dynamic-graph-service.info"`   |
| `frontend.limitConnections`  | The number of concurrent connections allowed from a single IP address    | `10`                             |

### Common Pod parameters

All workers have the same parameters of common pod assignment, worker-type={coordinator, sampling, serving}.

| Name                                                 | Description                                                                                                | Value                               |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `${worker-type}.updateStrategy.type`                 | Pod deployment strategy type                                                                               | `"RollingUpdate"`                   |
| `${worker-type}.updateStrategy.rollingUpdate`        | Pod deployment rolling update configuration parameters                                                     | `{}`                                |
| `${worker-type}.podLabels`                           | Extra labels for worker pod                                                                                | `{}`                                |
| `${worker-type}.podAnnotations`                      | Extra annotations for worker pod                                                                           | `{}`                                |
| `${worker-type}.podAffinityPreset`                   | Pod affinity preset. Ignored if `${worker-type}.affinity` is set. Allowed values: `soft` or `hard`         | `""`                                |
| `${worker-type}.podAntiAffinityPreset`               | Pod anti-affinity preset. Ignored if `${worker-type}.affinity` is set. Allowed values: `soft` or `hard`    | `"soft"`                            |
| `${worker-type}.nodeAffinityPreset.type`             | Node affinity preset type. Ignored if `${worker-type}.affinity` is set. Allowed values: `soft` or `hard`   | `""`                                |
| `${worker-type}.nodeAffinityPreset.key`              | Node label key to match Ignored if `${worker-type}.affinity` is set.                                       | `""`                                |
| `${worker-type}.nodeAffinityPreset.values`           | Node label values to match. Ignored if `${worker-type}.affinity` is set.                                   | `[]`                                |
| `${worker-type}.affinity`                            | Affinity for pod assignment                                                                                | `{}`                                |
| `${worker-type}.nodeSelector`                        | Node labels for pod assignment                                                                             | `{}`                                |
| `${worker-type}.tolerations`                         | Toleration for pod assignment                                                                              | `[]`                                |
| `${worker-type}.resources.limits`                    | The resources limits for the container                                                                     | `{}`                                |
| `${worker-type}.resources.requests`                  | The requested resources for the container                                                                  | `{}`                                |
| `${worker-type}.persistence.enabled`                 | Enable worker checkpoints persistence using PVC                                                            | `false`                             |
| `${worker-type}.persistence.storageClass`            | PVC Storage Class for checkpoint data volume                                                               | `""`                                |
| `${worker-type}.persistence.accessModes`             | Persistent Volume Access Modes                                                                             | `["ReadWriteOnce"]`                 |
| `${worker-type}.persistence.size`                    | PVC Storage Request for checkpoint data volume                                                             | `20Gi`                              |
| `${worker-type}.persistence.annotations`             | Annotations for the PVC                                                                                    | `{}`                                |
| `${worker-type}.persistence.selector`                | Selector to match an existing Persistent Volume for checkpoint data PVC.                                   | `{}`                                |
| `${worker-type}.persistence.mountPath`               | Mount path of the checkpoint data volume                                                                   | `"/${worker-type}_checkpoints"`     |
| `${worker-type}.livenessProbe.enabled`               | Enable livenessProbe on ${worker-type} containers                                                          | `true`                              |
| `${worker-type}.livenessProbe.initialDelaySeconds`   | Initial delay seconds for livenessProbe                                                                    | `10`                                |
| `${worker-type}.livenessProbe.periodSeconds`         | Period seconds for livenessProbe                                                                           | `10`                                |
| `${worker-type}.livenessProbe.timeoutSeconds`        | Timeout seconds for livenessProbe                                                                          | `1`                                 |
| `${worker-type}.livenessProbe.failureThreshold`      | Failure threshold for livenessProbe                                                                        | `3`                                 |
| `${worker-type}.livenessProbe.successThreshold`      | Success threshold for livenessProbe                                                                        | `1`                                 |

### Coordinator parameters

| Name                                              | Description                                                                                             | Value                               |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `coordinator.readinessProbe.enabled`              | Enable readinessProbe on coordinator containers                                                         | `true`                              |
| `coordinator.readinessProbe.initialDelaySeconds`  | Initial delay seconds for readinessProbe                                                                | `5`                                 |
| `coordinator.readinessProbe.periodSeconds`        | Period seconds for readinessProbe                                                                       | `10`                                |
| `coordinator.readinessProbe.timeoutSeconds`       | Timeout seconds for readinessProbe                                                                      | `1`                                 |
| `coordinator.readinessProbe.failureThreshold`     | Failure threshold for readinessProbe                                                                    | `6`                                 |
| `coordinator.readinessProbe.successThreshold`     | Success threshold for readinessProbe                                                                    | `1`                                 |
| `coordinator.rpcService.port`                     | Coordinator headless rpc service port for internal connections                                          | `50051`                             |
| `coordinator.rpcService.clusterIP`                | Static clusterIP or None for Coordinator headless rpc service                                           | `""`                                |
| `coordinator.rpcService.sessionAffinity`          | Control where internal rpc requests go, to the same pod or round-robin                                  | `None`                              |
| `coordinator.rpcService.annotations`              | Additional custom annotations for Coordinator headless rpc service                                      | `{}`                                |
| `coordinator.httpService.port`                    | Coordinator http service port for external admin requests                                               | `8080`                              |
| `coordinator.httpService.sessionAffinity`         | Control where external http requests go, to the same pod or round-robin                                 | `None`                              |
| `coordinator.httpService.externalTrafficPolicy`   | Coordinator http service external traffic policy                                                        | `Cluster`                           |
| `coordinator.httpService.annotations`             | Additional custom annotations for Coordinator http service                                              | `{}`                                |
| `coordinator.workdir`                             | Local ephemeral storage mount path for Coordinator working directory                                    | `"/coordinator_workdir"`            |
| `coordinator.connectTimeoutSeconds`               | The max timeout seconds when other workers connect to Coordinator                                       | `60`                                |
| `coordinator.heartbeatIntervalSeconds`            | The heartbeat interval in seconds when other workers report statistics to coordinator                   | `10`                                |

### Sampling parameters

| Name                                               | Description                                                                                          | Value                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| `sampling.workerNum`                               | Number of Sampling workers                                                                           | `2`                              |
| `sampling.workdir`                                 | Local ephemeral storage mount path for Sampling working directory                                    | `"/sampling_workdir"`            |
| `sampling.actorLocalShardNum`                      | Local computing shard number for each Sampling Worker pod                                            | `4`                              |
| `sampling.dataPartitionNum`                        | The total partition number of data across all Sampling Workers                                       | `8`                              |
| `sampling.rocksdbEnv.highPriorityThreads`          | The thread number of high-priority rocksdb background tasks                                          | `2`                              |
| `sampling.rocksdbEnv.lowPriorityThreads`           | The thread number of low-priority rocksdb background tasks                                           | `2`                              |
| `sampling.sampleStore.memtableRep`                 | The rocksdb memtable structure type of sample store                                                  | `"hashskiplist"`                 |
| `sampling.sampleStore.hashBucketCount`             | The hash bucket count of sample store memtable                                                       | `1048576`                        |
| `sampling.sampleStore.skipListLookahead`           | The look-ahead factor of sample store memtable                                                       | `0`                              |
| `sampling.sampleStore.blockCacheCapacity`          | The capacity (bytes) of sample store block cache                                                     | `67108864`                       |
| `sampling.sampleStore.ttlHours`                    | The TTL hours for sampling data in sample store                                                      | `1200`                           |
| `sampling.subscriptionTable.memtableRep`           | The rocksdb memtable structure type of Sampling subscription table                                   | `"hashskiplist"`                 |
| `sampling.subscriptionTable.hashBucketCount`       | The hash bucket count of subscription table memtable                                                 | `1048576`                        |
| `sampling.subscriptionTable.skipListLookahead`     | The look-ahead factor of subscription table memtable                                                 | `0`                              |
| `sampling.subscriptionTable.blockCacheCapacity`    | The capacity (bytes) of subscription table block cache                                               | `67108864`                       |
| `sampling.subscriptionTable.ttlHours`              | The TTL hours for sampling rules in subscription table                                               | `1200`                           |
| `sampling.recordPolling.threadNum`                 | The thread number for graph update consuming from kafka queues                                       | `2`                              |
| `sampling.recordPolling.retryIntervalMs`           | The retry interval (ms) when no record has been polled                                               | `100`                            |
| `sampling.recordPolling.processConcurrency`        | The max processing concurrency for polled records                                                    | `100`                            |
| `sampling.samplePublishing.producerPoolSize`       | The max number of kafka producer for sampling results                                                | `2`                              |
| `sampling.samplePublishing.maxProduceRetryTimes`   | The maximum retry times of producing a kafka message                                                 | `3`                              |
| `sampling.samplePublishing.callbackPollIntervalMs` | The interval(ms) for polling async producing callbacks                                               | `100`                            |
| `sampling.logging.dataLogPeriod`                   | Specify how many graph update batches should be processed between two logs                           | `10`                             |
| `sampling.logging.ruleLogPeriod`                   | Specify how many sampling rules should be processed between two logs                                 | `10`                             |

### Serving parameters

| Name                                              | Description                                                                                         | Value                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------- |
| `serving.workerNum`                               | Number of Serving workers, each Serving Worker is an independent pod                                | `2`                              |
| `serving.readinessProbe.enabled`                  | Enable readinessProbe on Serving worker containers                                                  | `true`                           |
| `serving.readinessProbe.initialDelaySeconds`      | Initial delay seconds for readinessProbe                                                            | `30`                             |
| `serving.readinessProbe.periodSeconds`            | Period seconds for readinessProbe                                                                   | `10`                             |
| `serving.readinessProbe.timeoutSeconds`           | Timeout seconds for readinessProbe                                                                  | `1`                              |
| `serving.readinessProbe.failureThreshold`         | Failure threshold for readinessProbe                                                                | `6`                              |
| `serving.readinessProbe.successThreshold`         | Success threshold for readinessProbe                                                                | `1`                              |
| `serving.httpService.port`                        | The external port of Serving http service for inference queries                                     | `10000`                          |
| `serving.httpService.sessionAffinity`             | Control where http requests go, to the same pod or round-robin                                      | `None`                           |
| `serving.httpService.externalTrafficPolicy`       | The external traffic policy of Serving http service                                                 | `Cluster`                        |       
| `serving.httpService.annotations`                 | Additional custom annotations of Serving http service                                               | `{}`                             |
| `serving.workdir`                                 | Local ephemeral storage mount path for Serving working directory                                    | `"/serving_workdir"`             |
| `serving.actorLocalShardNum`                      | Local computing shard number for each Serving Worker pod                                            | `4`                              |
| `serving.dataPartitionNum`                        | The partition number of data for each Serving Worker                                                | `4`                              |
| `serving.rocksdbEnv.highPriorityThreads`          | The thread number of high-priority rocksdb background tasks                                         | `2`                              |
| `serving.rocksdbEnv.lowPriorityThreads`           | The thread number of low-priority rocksdb background tasks                                          | `2`                              |
| `serving.sampleStore.inMemoryMode`                | Specify whether to open rocksdb in-memory mode of sample store                                      | `false`
| `serving.sampleStore.memtableRep`                 | The rocksdb memtable structure type of sample store                                                 | `"hashskiplist"`                 |
| `serving.sampleStore.hashBucketCount`             | The hash bucket count of sample store memtable                                                      | `1048576`                        |
| `serving.sampleStore.skipListLookahead`           | The look-ahead factor of sample store memtable                                                      | `0`                              |
| `serving.sampleStore.blockCacheCapacity`          | The capacity (bytes) of sample store block cache                                                    | `67108864`                       |
| `serving.sampleStore.ttlHours`                    | The TTL hours for serving data in sample store                                                      | `1200`                           |
| `serving.recordPolling.threadNum`                 | The thread number for sample update consuming from kafka queues                                     | `2`                              |
| `serving.recordPolling.retryIntervalMs`           | The retry interval (ms) when no record has been polled                                              | `100`                            |
| `serving.recordPolling.processConcurrency`        | The max processing concurrency for polled records                                                   | `100`                            |
| `serving.logging.dataLogPeriod`                   | Specify how many sample update batches should be processed between two logs                         | `10`                             |
| `serving.logging.requestLogPeriod`                | Interval of incoming inference query requests for logging serving statistics                        | `1`                              |

### Other Parameters

| Name                                          | Description                                                                                    | Value   |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------- |
| `serviceAccount.create`                       | Enable creation of ServiceAccount for pods                                                     | `true`  |
| `serviceAccount.name`                         | The name of the service account to use. If not set and `create` is `true`, a name is generated | `""`    |
| `serviceAccount.automountServiceAccountToken` | Allows auto mount of ServiceAccountToken on the serviceAccount created                         | `true`  |
| `serviceAccount.annotations`                  | Additional custom annotations for the ServiceAccount                                           | `{}`    |
| `rbac.create`                                 | Whether to create & use RBAC resources or not                                                  | `false` |