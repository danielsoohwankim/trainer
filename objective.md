repo: https://github.com/kubeflow/trainer
- gpu is one constraint
- every scenario, optiimize gpu usage, inlcuding runtime 
- terminate training jobs that are no longer making progress
- investigate this repo, training lifecycles, mechanisms, implement the minimal version of this feature to determine if we can determine jobs that are no longer making progress

start with core API 
unit tests