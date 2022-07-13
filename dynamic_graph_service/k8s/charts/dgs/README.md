## Introduction

This chart bootstraps a Dynamic-Graph-Service deployment on a [Kubernetes](https://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure
- Pre-deployed [Kafka](https://kafka.apache.org/) queue service with corresponding topics created, configure it with [Kafka service parameters](#kafka-service-parameters)

## Installing the Chart

To install the chart with the release name `my-release`(TODO: support remote helm repo):

```console
helm install my-release $project_dir/k8s/charts/dgs \
    --set-file graphSchema=/path/to/schema/json/file
```

The graph schema must be specified from a json string or file by parameter `graphSchema`.
A [template](../../../conf/schema.template.json) schema file can be followed to write your customized graph schema.

These commands deploy Dynamic-Graph-Service on the Kubernetes cluster in the default configuration.
The [Parameters](#parameters) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Uninstalling the Chart

To uninstall/delete the `my-release` deployment:

```console
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
| `labelGroupOverride`       | The self-managed label group prefix name                                                | `""`                  |
| `remoteFileRepoURL`        | The url of remote file repository that contains service binary packages and datasets    | `"https://graphlearn.oss-cn-hangzhou.aliyuncs.com"` |
| `graphSchema`              | The json string of graph schema, **must** be set during installation                    | `""`                  |
| `configPath`               | The service configmap mount path                                                        | `"/dgs_conf"`         |
| `glog.toConsole`           | Specify whether program logs are written to standard error as well as to files          | `false`               |

### Kafka service parameters

| Name                        | Description                                                                   | Value                |
| --------------------------- | ----------------------------------------------------------------------------- | -------------------- |
| `kafka.dl2spl.brokers`      | Kafka brokers of processed graph updates from dataloader to sampling workers  | `["localhost:9092"]` |
| `kafka.dl2spl.topic`        | Kafka topic of (dataloader -> sampling workers)                               | `"record-batches"`   |
| `kafka.dl2spl.partitions`   | Topic partition number of (dataloader -> sampling workers)                    | `2`                  |
| `kafka.spl2srv.brokers`     | Kafka brokers of sampled updates from sampling workers to serving workers     | `["localhost:9092"]` |
| `kafka.spl2srv.topic`       | Kafka topic of (sampling workers -> serving workers)                          | `"sample-batches"`   |
| `kafka.spl2srv.partitions`  | Topic partition number of (sampling workers -> serving workers)               | `4`                  |

### Image parameters

| Name                            | Description                                                            | Value                |
| ------------------------------- | ---------------------------------------------------------------------- | -------------------- |
| `image.dgs.registry`            | Core service image registry                                            | `"graphlearn"`       |
| `image.dgs.repository`          | Core service image repository                                          | `"dgs-core"`         |
| `image.dgs.tag`                 | Core service image tag (immutable tags are recommended)                | `"1.0.0"`            |
| `image.dgs.pullPolicy`          | Core service image pull policy                                         | `IfNotPresent`       |
| `image.dgs.pullSecrets`         | Specify docker-registry secret names as an array                       | `[]`                 |
| `image.dl.registry`             | Dataloader image registry                                              | `"graphlearn"`       |
| `image.dl.repository`           | Dataloader image repository                                            | `"dgs-dl"`           |
| `image.dl.tag`                  | Dataloader image tag (immutable tags are recommended)                  | `"1.0.0"`            |
| `image.dl.pullPolicy`           | Dataloader image pull policy                                           | `IfNotPresent`       |
| `image.dl.pullSecrets`          | Specify docker-registry secret names as an array                       | `[]`                 |

### FrontEnd parameters

| Name                         | Description                                                              | Value                            |
| ---------------------------- | ------------------------------------------------------------------------ | -------------------------------- |
| `frontend.ingressHostName`   | The host name of external ingress                                        | `"dynamic-graph-service.info"`   |
| `frontend.limitConnections`  | The number of concurrent connections allowed from a single IP address    | `10`                             |

### Common Pod parameters

All workers have the same parameters of common pod assignment, worker-type={coordinator, dataloader, sampling, serving}.

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
| `coordinator.heartbeatIntervalSeconds`            | The heartbeat interval in seconds when other workers report statistics to coordinator                   | `60`                                |

### Dataloader parameters

| Name                                             | Description                                                                                            | Value                               |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| `dataloader.replicaCount`                        | Number of Dataloader workers                                                                           | `1`                                 |
| `dataloader.workdir`                             | Local ephemeral storage mount path for Dataloader working directory                                    | `"/dataloader_workdir"`             |
| `dataloader.outputBatchSize`                     | The max number of graph update records in one output record batch                                      | `16`                                |
| `dataloader.sourceType`                          | The data source type of dataloader. Allowed values: `dblp` or `graphscope`                             | `"dblp"`                            |

Specify the following additional parameters when loading from [GraphScope-Store](https://graphscope.io/docs/persistent_graph_store.html) service:

| Name                                                       | Description                                                          | Value                          |
| ---------------------------------------------------------- | -------------------------------------------------------------------  | ------------------------------ |
| `dataloader.graphscope.logPolling.kafkaBrokers`            | The brokers of kafka service for polling                             | `["localhost:9092"]`           |
| `dataloader.graphscope.logPolling.kafkaTopic`              | The kafka topic name of graph update logs                            | `"graph-store"`                |
| `dataloader.graphscope.logPolling.kafkaPartitions`         | The partition number of polling kafka topic                          | `2`                            |
| `dataloader.graphscope.logPolling.offsetPersistIntervalMs` | The interval(ms) for persisting current log polling progress         | `5000`                         |
| `dataloader.graphscope.logPolling.retryIntervalMs`         | The interval(ms) for retries after an invalid polling                | `100`                          |
| `dataloader.graphscope.logPolling.flushIntervalMs`         | The max latency(ms) to flush current polled data into output queues  | `100`                          |
| `dataloader.graphscope.bulkLoading.threadNum`              | The number of threads for bulk loading                               | `2`                            |

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
| `sampling.logging.dataLogPeriod`                   | Specify how many graph update batches should be processed between two logs produced                  | `1`                              |
| `sampling.logging.ruleLogPeriod`                   | Specify how many sampling rules should be processed between two logs produced                        | `1`                              |

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
| `serving.logging.dataLogPeriod`                   | Specify how many sample update batches should be processed between two logs produced                | `1`                              |
| `serving.logging.requestLogPeriod`                | Interval of incoming inference query requests for logging serving statistics                        | `1`                              |

### Other Parameters

| Name                                          | Description                                                                                    | Value   |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------- |
| `serviceAccount.create`                       | Enable creation of ServiceAccount for pods                                                     | `true`  |
| `serviceAccount.name`                         | The name of the service account to use. If not set and `create` is `true`, a name is generated | `""`    |
| `serviceAccount.automountServiceAccountToken` | Allows auto mount of ServiceAccountToken on the serviceAccount created                         | `true`  |
| `serviceAccount.annotations`                  | Additional custom annotations for the ServiceAccount                                           | `{}`    |
| `rbac.create`                                 | Whether to create & use RBAC resources or not                                                  | `false` |