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
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/v2/pkg/constants"
	utiltesting "github.com/kubeflow/trainer/v2/pkg/util/testing"
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

func TestFindStalledTrainJobs(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))

	progressingTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "progressing"},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   recentUpdate,
			},
		},
	}
	stuckTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "stuck"},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   staleUpdate,
			},
		},
	}
	unresponsiveTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "unresponsive"},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    staleUpdate,
			},
		},
	}
	completedTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "completed"},
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
	}

	cases := map[string]struct {
		trainJobs []trainer.TrainJob
		want      []string
	}{
		// With no TrainJobs, no TrainJob is stalled.
		"no TrainJobs": {
			trainJobs: nil,
			want:      nil,
		},
		// TrainJobs that are progressing or finished are not returned.
		"no stalled TrainJobs": {
			trainJobs: []trainer.TrainJob{progressingTrainJob, completedTrainJob},
			want:      nil,
		},
		// Only the stuck and unresponsive TrainJobs are returned, in their original order.
		"some stalled TrainJobs": {
			trainJobs: []trainer.TrainJob{stuckTrainJob, progressingTrainJob, unresponsiveTrainJob, completedTrainJob},
			want:      []string{"stuck", "unresponsive"},
		},
		// Every TrainJob is returned when all of them are stalled.
		"all stalled TrainJobs": {
			trainJobs: []trainer.TrainJob{unresponsiveTrainJob, stuckTrainJob},
			want:      []string{"unresponsive", "stuck"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var got []string
			for _, trainJob := range FindStalledTrainJobs(tc.trainJobs, now, progressDeadline) {
				got = append(got, trainJob.Name)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Unexpected stalled TrainJobs (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestStopStalledTrainJobs(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))
	errInjected := errors.New("injected error")

	progressingTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "progressing", Namespace: metav1.NamespaceDefault},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   recentUpdate,
			},
		},
	}
	stuckTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "stuck", Namespace: metav1.NamespaceDefault},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   staleUpdate,
			},
		},
	}
	unresponsiveTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "unresponsive", Namespace: metav1.NamespaceDefault},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    staleUpdate,
			},
		},
	}
	externallyManagedTrainJob := trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "externally-managed", Namespace: metav1.NamespaceDefault},
		Spec: trainer.TrainJobSpec{
			ManagedBy: ptr.To("kueue.x-k8s.io/multikueue"),
		},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    staleUpdate,
			},
		},
	}

	cases := map[string]struct {
		trainJobs []trainer.TrainJob
		// jobSets are the names of the JobSets that exist before StopStalledTrainJobs is called.
		jobSets []string
		// statusPatchErrFor and jobSetDeleteErrFor are the names of the TrainJobs for which
		// the status patch and the JobSet deletion fail.
		statusPatchErrFor  string
		jobSetDeleteErrFor string
		wantStopped        []string
		wantFailed         []string
		wantJobSets        []string
		wantErr            bool
	}{
		// A stalled TrainJob is marked as Failed and its JobSet is deleted, so its GPUs are released.
		"stalled TrainJob is stopped": {
			trainJobs:   []trainer.TrainJob{stuckTrainJob},
			jobSets:     []string{"stuck"},
			wantStopped: []string{"stuck"},
			wantFailed:  []string{"stuck"},
			wantJobSets: nil,
		},
		// A TrainJob that is making progress is left running.
		"progressing TrainJob is not stopped": {
			trainJobs:   []trainer.TrainJob{progressingTrainJob},
			jobSets:     []string{"progressing"},
			wantStopped: nil,
			wantFailed:  nil,
			wantJobSets: []string{"progressing"},
		},
		// Only the stuck and unresponsive TrainJobs are stopped, in their original order.
		"only stalled TrainJobs are stopped": {
			trainJobs:   []trainer.TrainJob{stuckTrainJob, progressingTrainJob, unresponsiveTrainJob},
			jobSets:     []string{"stuck", "progressing", "unresponsive"},
			wantStopped: []string{"stuck", "unresponsive"},
			wantFailed:  []string{"stuck", "unresponsive"},
			wantJobSets: []string{"progressing"},
		},
		// A stalled TrainJob whose JobSet is already gone is still marked as Failed without an error.
		"stalled TrainJob without JobSet is stopped": {
			trainJobs:   []trainer.TrainJob{stuckTrainJob},
			jobSets:     nil,
			wantStopped: []string{"stuck"},
			wantFailed:  []string{"stuck"},
			wantJobSets: nil,
		},
		// A stalled TrainJob managed by an external controller (e.g. MultiKueue) is left to that controller.
		"stalled TrainJob managed by an external controller is not stopped": {
			trainJobs:   []trainer.TrainJob{externallyManagedTrainJob},
			jobSets:     []string{"externally-managed"},
			wantStopped: nil,
			wantFailed:  nil,
			wantJobSets: []string{"externally-managed"},
		},
		// When marking a TrainJob as Failed fails, its JobSet is kept so the TrainJob controller
		// doesn't recreate it, and the other stalled TrainJobs are still stopped.
		"stalled TrainJob is not stopped when marking it as failed fails": {
			trainJobs:         []trainer.TrainJob{stuckTrainJob, unresponsiveTrainJob},
			jobSets:           []string{"stuck", "unresponsive"},
			statusPatchErrFor: "stuck",
			wantStopped:       []string{"unresponsive"},
			wantFailed:        []string{"unresponsive"},
			wantJobSets:       []string{"stuck"},
			wantErr:           true,
		},
		// When deleting the JobSet fails, the TrainJob stays marked as Failed but is not reported as stopped.
		"stalled TrainJob is not stopped when deleting its JobSet fails": {
			trainJobs:          []trainer.TrainJob{stuckTrainJob},
			jobSets:            []string{"stuck"},
			jobSetDeleteErrFor: "stuck",
			wantStopped:        nil,
			wantFailed:         []string{"stuck"},
			wantJobSets:        []string{"stuck"},
			wantErr:            true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			var objs []client.Object
			for i := range tc.trainJobs {
				objs = append(objs, tc.trainJobs[i].DeepCopy())
			}
			for _, jobSetName := range tc.jobSets {
				objs = append(objs, &jobsetv1alpha2.JobSet{
					ObjectMeta: metav1.ObjectMeta{Name: jobSetName, Namespace: metav1.NamespaceDefault},
				})
			}
			cli := utiltesting.NewClientBuilder().
				WithObjects(objs...).
				WithStatusSubresource(&trainer.TrainJob{}).
				WithInterceptorFuncs(interceptor.Funcs{
					SubResourcePatch: func(ctx context.Context, cli client.Client, subResourceName string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
						if obj.GetName() == tc.statusPatchErrFor {
							return errInjected
						}
						return cli.SubResource(subResourceName).Patch(ctx, obj, patch, opts...)
					},
					Delete: func(ctx context.Context, cli client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
						if _, ok := obj.(*jobsetv1alpha2.JobSet); ok && obj.GetName() == tc.jobSetDeleteErrFor {
							return errInjected
						}
						return cli.Delete(ctx, obj, opts...)
					},
				}).
				Build()

			trainJobs := make([]trainer.TrainJob, len(tc.trainJobs))
			for i := range tc.trainJobs {
				tc.trainJobs[i].DeepCopyInto(&trainJobs[i])
			}
			stopped, err := StopStalledTrainJobs(ctx, cli, trainJobs, now, progressDeadline)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("StopStalledTrainJobs() error = %v, wantErr %v", err, tc.wantErr)
			}

			var gotStopped []string
			for _, trainJob := range stopped {
				gotStopped = append(gotStopped, trainJob.Name)
			}
			if diff := cmp.Diff(tc.wantStopped, gotStopped); diff != "" {
				t.Errorf("Unexpected stopped TrainJobs (-want,+got):\n%s", diff)
			}

			var gotFailed []string
			for _, trainJob := range tc.trainJobs {
				var got trainer.TrainJob
				if err := cli.Get(ctx, client.ObjectKeyFromObject(&trainJob), &got); err != nil {
					t.Fatalf("Unexpected error getting TrainJob %s: %v", trainJob.Name, err)
				}
				cond := meta.FindStatusCondition(got.Status.Conditions, trainer.TrainJobFailed)
				if cond != nil && cond.Status == metav1.ConditionTrue &&
					cond.Reason == trainer.TrainJobProgressDeadlineExceededReason &&
					cond.Message == constants.TrainJobProgressDeadlineExceededMessage {
					gotFailed = append(gotFailed, got.Name)
				}
			}
			if diff := cmp.Diff(tc.wantFailed, gotFailed); diff != "" {
				t.Errorf("Unexpected TrainJobs marked as failed (-want,+got):\n%s", diff)
			}

			var jobSets jobsetv1alpha2.JobSetList
			if err := cli.List(ctx, &jobSets); err != nil {
				t.Fatalf("Unexpected error listing JobSets: %v", err)
			}
			var gotJobSets []string
			for _, jobSet := range jobSets.Items {
				gotJobSets = append(gotJobSets, jobSet.Name)
			}
			slices.Sort(gotJobSets)
			wantJobSets := slices.Clone(tc.wantJobSets)
			slices.Sort(wantJobSets)
			if diff := cmp.Diff(wantJobSets, gotJobSets); diff != "" {
				t.Errorf("Unexpected remaining JobSets (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestGetTrainJobState(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))

	cases := map[string]struct {
		trainJob *trainer.TrainJob
		want     TrainJobState
	}{
		// A TrainJob that has not reported trainerStatus yet is active.
		"TrainJob without trainerStatus is active": {
			trainJob: &trainer.TrainJob{},
			want:     TrainJobStateActive,
		},
		// A TrainJob whose progress advanced within the progress deadline is active.
		"progressing TrainJob is active": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   recentUpdate,
					},
				},
			},
			want: TrainJobStateActive,
		},
		// A TrainJob stuck at the same progress beyond the progress deadline is stalled.
		"stuck TrainJob is stalled": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					TrainerStatus: &trainer.TrainerStatus{
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			want: TrainJobStateStalled,
		},
		// A TrainJob failed with the ProgressDeadlineExceeded reason was stopped by StopStalledTrainJobs.
		"TrainJob stopped by StopStalledTrainJobs is killed": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobFailed,
							Status: metav1.ConditionTrue,
							Reason: trainer.TrainJobProgressDeadlineExceededReason,
						},
					},
					TrainerStatus: &trainer.TrainerStatus{
						LastUpdatedTime: staleUpdate,
					},
				},
			},
			want: TrainJobStateKilled,
		},
		// A TrainJob failed for another reason, e.g. exceeding its active deadline, is failed.
		"TrainJob failed for another reason is failed": {
			trainJob: &trainer.TrainJob{
				Status: trainer.TrainJobStatus{
					Conditions: []metav1.Condition{
						{
							Type:   trainer.TrainJobFailed,
							Status: metav1.ConditionTrue,
							Reason: trainer.TrainJobDeadlineExceededReason,
						},
					},
				},
			},
			want: TrainJobStateFailed,
		},
		// A completed TrainJob is completed, even if its last update is stale.
		"completed TrainJob is completed": {
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
			want: TrainJobStateCompleted,
		},
		// A suspended TrainJob is suspended, even if its last update is stale.
		"suspended TrainJob is suspended": {
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
			want: TrainJobStateSuspended,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := GetTrainJobState(tc.trainJob, now, progressDeadline)
			if got != tc.want {
				t.Errorf("GetTrainJobState(%v, %v, %v) = %v, want %v", tc.trainJob, now, progressDeadline, got, tc.want)
			}
		})
	}
}

func TestNewTrainJobsReport(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))

	cases := map[string]struct {
		trainJobs []trainer.TrainJob
		want      *TrainJobsReport
	}{
		// With no TrainJobs, every state is reported with a count of zero.
		"no TrainJobs": {
			trainJobs: nil,
			want: &TrainJobsReport{
				GeneratedAt:      now,
				ProgressDeadline: progressDeadline,
				Counts: map[TrainJobState]int{
					TrainJobStateActive:    0,
					TrainJobStateStalled:   0,
					TrainJobStateKilled:    0,
					TrainJobStateCompleted: 0,
					TrainJobStateFailed:    0,
					TrainJobStateSuspended: 0,
				},
			},
		},
		// TrainJobs are counted per state, and reported sorted by namespace and name with their progress.
		"TrainJobs are counted per state and sorted by namespace and name": {
			trainJobs: []trainer.TrainJob{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "stuck", Namespace: "team-a"},
					Status: trainer.TrainJobStatus{
						TrainerStatus: &trainer.TrainerStatus{
							ProgressPercentage: ptr.To[int32](0),
							LastUpdatedTime:    recentUpdate,
							LastProgressTime:   staleUpdate,
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "progressing", Namespace: "default"},
					Status: trainer.TrainJobStatus{
						TrainerStatus: &trainer.TrainerStatus{
							ProgressPercentage: ptr.To[int32](50),
							LastUpdatedTime:    recentUpdate,
							LastProgressTime:   recentUpdate,
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "starting", Namespace: "team-a"},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "killed", Namespace: "default"},
					Status: trainer.TrainJobStatus{
						Conditions: []metav1.Condition{
							{
								Type:   trainer.TrainJobFailed,
								Status: metav1.ConditionTrue,
								Reason: trainer.TrainJobProgressDeadlineExceededReason,
							},
						},
						TrainerStatus: &trainer.TrainerStatus{
							ProgressPercentage: ptr.To[int32](40),
							LastUpdatedTime:    staleUpdate,
						},
					},
				},
			},
			want: &TrainJobsReport{
				GeneratedAt:      now,
				ProgressDeadline: progressDeadline,
				Counts: map[TrainJobState]int{
					TrainJobStateActive:    2,
					TrainJobStateStalled:   1,
					TrainJobStateKilled:    1,
					TrainJobStateCompleted: 0,
					TrainJobStateFailed:    0,
					TrainJobStateSuspended: 0,
				},
				TrainJobs: []TrainJobReport{
					{
						Namespace:          "default",
						Name:               "killed",
						State:              TrainJobStateKilled,
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    staleUpdate,
					},
					{
						Namespace:          "default",
						Name:               "progressing",
						State:              TrainJobStateActive,
						ProgressPercentage: ptr.To[int32](50),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   recentUpdate,
					},
					{
						Namespace: "team-a",
						Name:      "starting",
						State:     TrainJobStateActive,
					},
					{
						Namespace:          "team-a",
						Name:               "stuck",
						State:              TrainJobStateStalled,
						ProgressPercentage: ptr.To[int32](0),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := NewTrainJobsReport(tc.trainJobs, now, progressDeadline)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Unexpected TrainJobs report (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestMonitorTrainJobs(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	progressDeadline := 30 * time.Minute
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))
	errInjected := errors.New("injected error")

	progressingTrainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "progressing", Namespace: metav1.NamespaceDefault},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](50),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   recentUpdate,
			},
		},
	}
	stuckTrainJob := &trainer.TrainJob{
		ObjectMeta: metav1.ObjectMeta{Name: "stuck", Namespace: "team-a"},
		Status: trainer.TrainJobStatus{
			TrainerStatus: &trainer.TrainerStatus{
				ProgressPercentage: ptr.To[int32](40),
				LastUpdatedTime:    recentUpdate,
				LastProgressTime:   staleUpdate,
			},
		},
	}

	cases := map[string]struct {
		listOpts      []client.ListOption
		listErr       error
		wantTrainJobs []string
		wantCounts    map[TrainJobState]int
		wantErr       bool
	}{
		// TrainJobs in every namespace are listed and reported.
		"TrainJobs in all namespaces are reported": {
			wantTrainJobs: []string{"default/progressing", "team-a/stuck"},
			wantCounts: map[TrainJobState]int{
				TrainJobStateActive:    1,
				TrainJobStateStalled:   1,
				TrainJobStateKilled:    0,
				TrainJobStateCompleted: 0,
				TrainJobStateFailed:    0,
				TrainJobStateSuspended: 0,
			},
		},
		// List options, e.g. a namespace, limit the TrainJobs that are reported.
		"only TrainJobs in the namespace are reported": {
			listOpts:      []client.ListOption{client.InNamespace("team-a")},
			wantTrainJobs: []string{"team-a/stuck"},
			wantCounts: map[TrainJobState]int{
				TrainJobStateActive:    0,
				TrainJobStateStalled:   1,
				TrainJobStateKilled:    0,
				TrainJobStateCompleted: 0,
				TrainJobStateFailed:    0,
				TrainJobStateSuspended: 0,
			},
		},
		// An error listing the TrainJobs is returned without a report.
		"error listing TrainJobs is returned": {
			listErr: errInjected,
			wantErr: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			cli := utiltesting.NewClientBuilder().
				WithObjects(progressingTrainJob.DeepCopy(), stuckTrainJob.DeepCopy()).
				WithInterceptorFuncs(interceptor.Funcs{
					List: func(ctx context.Context, cli client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
						if tc.listErr != nil {
							return tc.listErr
						}
						return cli.List(ctx, list, opts...)
					},
				}).
				Build()

			report, err := MonitorTrainJobs(ctx, cli, now, progressDeadline, tc.listOpts...)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("MonitorTrainJobs() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				if report != nil {
					t.Errorf("MonitorTrainJobs() report = %v, want nil", report)
				}
				return
			}

			var gotTrainJobs []string
			for _, trainJob := range report.TrainJobs {
				gotTrainJobs = append(gotTrainJobs, trainJob.Namespace+"/"+trainJob.Name)
			}
			if diff := cmp.Diff(tc.wantTrainJobs, gotTrainJobs); diff != "" {
				t.Errorf("Unexpected reported TrainJobs (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantCounts, report.Counts); diff != "" {
				t.Errorf("Unexpected TrainJob counts (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestTrainJobsReportString(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	recentUpdate := metav1.NewTime(now.Add(-5 * time.Minute))
	staleUpdate := metav1.NewTime(now.Add(-2 * time.Hour))

	cases := map[string]struct {
		report *TrainJobsReport
		want   string
	}{
		// With no TrainJobs, the report shows zero counts and an empty table.
		"no TrainJobs": {
			report: &TrainJobsReport{
				GeneratedAt:      now,
				ProgressDeadline: 30 * time.Minute,
			},
			want: `TrainJobs report at 2026-10-01T12:00:00Z (progress deadline: 30m0s)
Total: 0  Active: 0  Stalled: 0  Killed: 0  Completed: 0  Failed: 0  Suspended: 0

NAMESPACE  NAME  STATE  PROGRESS  LAST UPDATE  LAST PROGRESS
`,
		},
		// Each TrainJob is shown with its state, progress bar, and time since its last update and progress.
		"TrainJobs with progress bars": {
			report: &TrainJobsReport{
				GeneratedAt:      now,
				ProgressDeadline: 30 * time.Minute,
				Counts: map[TrainJobState]int{
					TrainJobStateActive:  2,
					TrainJobStateStalled: 1,
					TrainJobStateKilled:  1,
				},
				TrainJobs: []TrainJobReport{
					{
						Namespace:          "default",
						Name:               "killed",
						State:              TrainJobStateKilled,
						ProgressPercentage: ptr.To[int32](40),
						LastUpdatedTime:    staleUpdate,
						LastProgressTime:   staleUpdate,
					},
					{
						Namespace:          "default",
						Name:               "progressing",
						State:              TrainJobStateActive,
						ProgressPercentage: ptr.To[int32](50),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   recentUpdate,
					},
					{
						Namespace: "team-a",
						Name:      "starting",
						State:     TrainJobStateActive,
					},
					{
						Namespace:          "team-a",
						Name:               "stuck",
						State:              TrainJobStateStalled,
						ProgressPercentage: ptr.To[int32](0),
						LastUpdatedTime:    recentUpdate,
						LastProgressTime:   staleUpdate,
					},
				},
			},
			want: `TrainJobs report at 2026-10-01T12:00:00Z (progress deadline: 30m0s)
Total: 4  Active: 2  Stalled: 1  Killed: 1  Completed: 0  Failed: 0  Suspended: 0

NAMESPACE  NAME         STATE    PROGRESS                     LAST UPDATE  LAST PROGRESS
default    killed       Killed   [████████░░░░░░░░░░░░]  40%  2h           2h
default    progressing  Active   [██████████░░░░░░░░░░]  50%  5m           5m
team-a     starting     Active   -                            -            -
team-a     stuck        Stalled  [░░░░░░░░░░░░░░░░░░░░]   0%  5m           2h
`,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, tc.report.String()); diff != "" {
				t.Errorf("Unexpected TrainJobs report (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestProgressBar(t *testing.T) {
	cases := map[string]struct {
		progressPercentage *int32
		want               string
	}{
		// Unknown progress is shown as "-".
		"unknown progress": {
			progressPercentage: nil,
			want:               "-",
		},
		// 0% progress is an empty bar.
		"no progress": {
			progressPercentage: ptr.To[int32](0),
			want:               "[░░░░░░░░░░░░░░░░░░░░]   0%",
		},
		// Each character is 5%, so 33% rounds down to 6 filled characters.
		"partial progress rounds down": {
			progressPercentage: ptr.To[int32](33),
			want:               "[██████░░░░░░░░░░░░░░]  33%",
		},
		// 50% progress is a half-filled bar.
		"half progress": {
			progressPercentage: ptr.To[int32](50),
			want:               "[██████████░░░░░░░░░░]  50%",
		},
		// 100% progress is a full bar.
		"full progress": {
			progressPercentage: ptr.To[int32](100),
			want:               "[████████████████████] 100%",
		},
		// Progress above 100% is shown as 100%.
		"progress above 100% is capped": {
			progressPercentage: ptr.To[int32](150),
			want:               "[████████████████████] 100%",
		},
		// Progress below 0% is shown as 0%.
		"progress below 0% is floored": {
			progressPercentage: ptr.To[int32](-10),
			want:               "[░░░░░░░░░░░░░░░░░░░░]   0%",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := progressBar(tc.progressPercentage)
			if got != tc.want {
				t.Errorf("progressBar(%v) = %q, want %q", ptr.Deref(tc.progressPercentage, -1), got, tc.want)
			}
		})
	}
}
