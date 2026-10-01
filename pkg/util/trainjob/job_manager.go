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
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
)

// StopStalledTrainJobs stops the stalled TrainJobs so that their resources (e.g. GPUs) are released,
// and returns the TrainJobs that were stopped. A stalled TrainJob is marked as Failed with the
// ProgressDeadlineExceeded reason before its JobSet is deleted, otherwise the TrainJob controller
// would recreate the JobSet. TrainJobs managed by an external controller are skipped.
func StopStalledTrainJobs(ctx context.Context, c client.Client, trainJobs []trainer.TrainJob, now time.Time, progressDeadline time.Duration) ([]*trainer.TrainJob, error) {
	log := ctrl.LoggerFrom(ctx)
	var stoppedTrainJobs []*trainer.TrainJob
	var errs error
	for _, trainJob := range FindStalledTrainJobs(trainJobs, now, progressDeadline) {
		if IsManagedByExternalController(trainJob) {
			continue
		}
		stoppedTrainJob := trainJob.DeepCopy()
		meta.SetStatusCondition(&stoppedTrainJob.Status.Conditions, metav1.Condition{
			Type:               trainer.TrainJobFailed,
			Status:             metav1.ConditionTrue,
			Reason:             trainer.TrainJobProgressDeadlineExceededReason,
			Message:            constants.TrainJobProgressDeadlineExceededMessage,
			LastTransitionTime: metav1.NewTime(now),
		})
		if err := c.Status().Patch(ctx, stoppedTrainJob, client.MergeFrom(trainJob)); err != nil {
			errs = errors.Join(errs, fmt.Errorf("failed to mark TrainJob %s as failed: %w", klog.KObj(trainJob), err))
			continue
		}
		jobSet := &jobsetv1alpha2.JobSet{
			ObjectMeta: metav1.ObjectMeta{Name: trainJob.Name, Namespace: trainJob.Namespace},
		}
		if err := client.IgnoreNotFound(c.Delete(ctx, jobSet)); err != nil {
			errs = errors.Join(errs, fmt.Errorf("failed to delete JobSet of TrainJob %s: %w", klog.KObj(trainJob), err))
			continue
		}
		log.V(2).Info("Stopped stalled TrainJob", "trainJob", klog.KObj(trainJob), "progressDeadline", progressDeadline)
		stoppedTrainJobs = append(stoppedTrainJobs, stoppedTrainJob)
	}
	return stoppedTrainJobs, errs
}

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
