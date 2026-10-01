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
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

func TestIsTrainJobStalled(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))

	cases := map[string]struct {
		trainJob         *trainer.TrainJob
		progressDeadline time.Duration
		want             bool
	}{
		// A TrainJob that has never reported progress is not treated as stalled.
		"TrainJob without trainerStatus is not stalled": {
			trainJob:         &trainer.TrainJob{},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob that last reported progress 5 minutes ago, within the 30-minute deadline, is not stalled.
		"TrainJob with trainerStatus updated within progress deadline is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob that last reported progress 2 hours ago, past the 30-minute deadline, is stalled.
		"TrainJob with trainerStatus not updated beyond progress deadline is stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             true,
		},
		// A TrainJob whose last update is exactly at the deadline boundary is not stalled, so the deadline is inclusive.
		"TrainJob with trainerStatus updated exactly at progress deadline is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: metav1.NewTime(now.Add(-progressDeadline)),
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob whose last update is in the future (clock skew) is not stalled.
		"TrainJob with trainerStatus updated in the future is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: metav1.NewTime(now.Add(10 * time.Minute)),
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob that reports 100% but then stops updating is still stalled, because only a Complete condition ends the check.
		"TrainJob at 100% progress not updated beyond progress deadline is stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](100),
						LastUpdatedTime:    staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             true,
		},
		// A TrainJob that still reports every few minutes but has been stuck at 40% for 2 hours, past the 30-minute deadline, is stalled.
		"TrainJob stuck at the same progress beyond progress deadline is stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             true,
		},
		// A TrainJob whose progress last advanced 20 minutes ago, within the 30-minute deadline, is not stalled.
		"TrainJob with progress advanced within progress deadline is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   metav1.NewTime(now.Add(-20 * time.Minute)),
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob whose progress last advanced exactly at the deadline boundary is not stalled, so the deadline is inclusive.
		"TrainJob stuck at the same progress exactly at progress deadline is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   metav1.NewTime(now.Add(-progressDeadline)),
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A TrainJob that reports regularly but never moves past 0% (e.g. hung while loading data) is stalled.
		"TrainJob stuck at 0% progress beyond progress deadline is stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](0),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             true,
		},
		// A TrainJob that sat at 100% for 2 hours (e.g. uploading the model) and then completed is not stalled.
		"completed TrainJob stuck at the same progress before completing is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobComplete,
							Status: metav1.ConditionTrue,
						},
					},
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](100),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A suspended TrainJob stuck at the same progress is not stalled, because it isn't expected to make progress while suspended.
		"suspended TrainJob stuck at the same progress is not stalled": {
			trainJob: &trainer.TrainJob{
				Spec: trainer.TrainJobSpec{
					Suspend: ptr.To(true),
				},
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A completed TrainJob is never stalled, even if its last update is stale.
		"completed TrainJob with stale trainerStatus is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobComplete,
							Status: metav1.ConditionTrue,
						},
					},
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A failed TrainJob is never stalled, even if its last update is stale.
		"failed TrainJob with stale trainerStatus is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobFailed,
							Status: metav1.ConditionTrue,
						},
					},
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A suspended TrainJob is not stalled, because it isn't expected to make progress while suspended.
		"suspended TrainJob with stale trainerStatus is not stalled": {
			trainJob: &trainer.TrainJob{
				Spec: trainer.TrainJobSpec{
					Suspend: ptr.To(true),
				},
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: staleUpdate,
					},
				},
			},
			progressDeadline: progressDeadline,
			want:             false,
		},
		// A progress deadline of zero turns stall detection off, so even a stale TrainJob is not stalled.
		"TrainJob with stale trainerStatus and zero progress deadline is not stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: staleUpdate,
					},
				},
			},
			progressDeadline: 0,
			want:             false,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := IsTrainJobStalled(tc.trainJob, now, tc.progressDeadline)
			if got != tc.want {
				t.Errorf("IsTrainJobStalled(%v, %v, %v) = %v, want %v", tc.trainJob, now, tc.progressDeadline, got, tc.want)
			}
		})
	}
}
