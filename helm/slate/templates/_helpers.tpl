# Modified: 2026-02-08T07:15:00Z | Author: COPILOT | Change: Create Helm helpers template
{{/*
SLATE chart helper definitions
*/}}

{{/* Full name */}}
{{- define "slate.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Common labels */}}
{{- define "slate.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/part-of: slate-system
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/* Selector labels */}}
{{- define "slate.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Namespace */}}
{{- define "slate.namespace" -}}
{{ .Values.global.namespace | default "slate" }}
{{- end }}
