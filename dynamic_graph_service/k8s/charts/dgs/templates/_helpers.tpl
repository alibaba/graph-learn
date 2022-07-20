{{/*
Expand the name of the chart.
*/}}
{{- define "dgs.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "dgs.fullname" -}}
{{- if .Values.fullnameOverride -}}
  {{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
  {{- $name := default .Chart.Name .Values.nameOverride -}}
  {{- if contains $name .Release.Name -}}
    {{- .Release.Name | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{- define "dgs.label.group" -}}
{{- if .Values.labelGroupOverride -}}
  {{- .Values.labelGroupOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
  {{- .Values.frontend.ingressHostName | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "dgs.configmap.name" -}}
{{- printf "%s-%s" (include "dgs.fullname" .) "config" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dgs.coordinator.name" -}}
{{- printf "%s-%s" (include "dgs.fullname" .) "coordinator" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dgs.coordinator.rpc.svc.host" -}}
{{- printf "%s-0.%s-headless.%s.svc.%s" (include "dgs.coordinator.name" .) (include "dgs.coordinator.name" .) .Release.Namespace .Values.clusterDomain -}}
{{- end -}}

{{- define "dgs.sampling.name" -}}
{{- printf "%s-%s" (include "dgs.fullname" .) "sampling" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dgs.serving.name" -}}
{{- printf "%s-%s" (include "dgs.fullname" .) "serving" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dgs.frontend.name" -}}
{{- printf "%s-%s" (include "dgs.fullname" .) "frontend" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "dgs.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "dgs.labels" -}}
helm.sh/chart: {{ include "dgs.chart" . }}
{{ include "dgs.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dgs.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dgs.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "dgs.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
  {{- default (include "dgs.fullname" .) .Values.serviceAccount.name }}
{{- else }}
  {{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
{{ include "dgs.images.image" (dict "imageRoot" .Values.path.to.the.image) }}
*/}}
{{- define "dgs.images.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $tag := .imageRoot.tag | toString -}}
{{- if $registryName }}
  {{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else -}}
  {{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end -}}

{{/*
Return the proper Docker Image Registry Secret Names
{{ include "dgs.images.pullSecrets" (dict "imageRoot" .Values.path.to.the.image) }}
*/}}
{{- define "dgs.images.pullSecrets" -}}
{{- $pullSecrets := list }}
{{- range .imageRoot.pullSecrets -}}
  {{- $pullSecrets = append $pullSecrets . -}}
{{- end -}}
{{- if (not (empty $pullSecrets)) }}
imagePullSecrets:
  {{- range $pullSecrets }}
  - name: {{ . }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
Return the proper Storage Class
*/}}
{{- define "dgs.storage.class" -}}
{{- if .storageClass -}}
  {{- if (eq "-" .storageClass) -}}
      {{- printf "storageClassName: \"\"" -}}
  {{- else }}
      {{- printf "storageClassName: %s" .storageClass -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{/*
Return the glog options
*/}}
{{- define "dgs.glog.option" -}}
{{- $logToConsole := .Values.glog.toConsole -}}
{{- if $logToConsole -}}
  {{- printf "--log-to-console" -}}
{{- end -}}
{{- end -}}

{{/*
Renders a value that contains template.
Usage:
{{ include "common.tplvalues.render" ( dict "value" .Values.path.to.the.Value "context" $) }}
*/}}
{{- define "common.tplvalues.render" -}}
{{- if typeIs "string" .value }}
  {{- tpl .value .context }}
{{- else }}
  {{- tpl (.value | toYaml) .context }}
{{- end }}
{{- end -}}
