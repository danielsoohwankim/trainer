/*
Copyright The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package trainjob

import (
	"time"

	"k8s.io/utils/ptr"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

// FindStalledTrainJobs checks the stalled status of every TrainJob and returns the stalled ones
// in their original order.
func FindStalledTrainJobs(trainJobs []trainer.TrainJob, now time.Time, progressDeadline time.Duration) []*trainer.TrainJob {
	var stalledTrainJobs []*trainer.TrainJob
	for i := range trainJobs {
		if IsTrainJobStalled(&trainJobs[i], now, progressDeadline) {
			stalledTrainJobs = append(stalledTrainJobs, &trainJobs[i])
		}
	}
	return stalledTrainJobs
}

// IsTrainJobStalled returns true when a running TrainJob is no longer making progress, so its
// resources (e.g. GPUs) can be reclaimed. A TrainJob is stalled when its trainer has either stopped
// reporting status (lastUpdatedTime) or stopped advancing progressPercentage (lastProgressTime)
// for longer than progressDeadline. Finished and suspended TrainJobs, TrainJobs that have not
// reported trainerStatus, and a non-positive progressDeadline are never considered stalled.
func IsTrainJobStalled(trainJob *trainer.TrainJob, now time.Time, progressDeadline time.Duration) bool {
	trainerStatus := trainJob.Status.TrainerStatus
	if progressDeadline <= 0 || trainerStatus == nil ||
		IsTrainJobFinished(trainJob) || ptr.Deref(trainJob.Spec.Suspend, false) {
		return false
	}
	lastProgressTime := trainerStatus.LastProgressTime
	return now.Sub(trainerStatus.LastUpdatedTime.Time) > progressDeadline ||
		(!lastProgressTime.IsZero() && now.Sub(lastProgressTime.Time) > progressDeadline)
}
