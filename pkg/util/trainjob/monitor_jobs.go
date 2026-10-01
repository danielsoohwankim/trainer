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
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/duration"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainer "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
)

// TrainJobState is the state of a TrainJob as reported by MonitorTrainJobs.
type TrainJobState string

const (
	// TrainJobStateActive means that the TrainJob is running and not stalled.
	TrainJobStateActive TrainJobState = "Active"

	// TrainJobStateStalled means that the TrainJob is no longer making progress and can be stopped.
	TrainJobStateStalled TrainJobState = "Stalled"

	// TrainJobStateKilled means that the TrainJob was stopped by StopStalledTrainJobs.
	TrainJobStateKilled TrainJobState = "Killed"

	// TrainJobStateCompleted means that the TrainJob has completed.
	TrainJobStateCompleted TrainJobState = "Completed"

	// TrainJobStateFailed means that the TrainJob has failed for a reason other than being stalled.
	TrainJobStateFailed TrainJobState = "Failed"

	// TrainJobStateSuspended means that the TrainJob is suspended.
	TrainJobStateSuspended TrainJobState = "Suspended"
)

// trainJobStates lists the TrainJob states in the order they are reported.
var trainJobStates = []TrainJobState{
	TrainJobStateActive,
	TrainJobStateStalled,
	TrainJobStateKilled,
	TrainJobStateCompleted,
	TrainJobStateFailed,
	TrainJobStateSuspended,
}

// progressBarWidth is the number of characters in a progress bar, so each one represents 5%.
const progressBarWidth = 20

// TrainJobReport is the reported state and progress of a single TrainJob.
type TrainJobReport struct {
	Namespace          string
	Name               string
	State              TrainJobState
	ProgressPercentage *int32
	LastUpdatedTime    metav1.Time
	LastProgressTime   metav1.Time
}

// TrainJobsReport is a snapshot of the state and progress of all TrainJobs.
type TrainJobsReport struct {
	GeneratedAt      time.Time
	ProgressDeadline time.Duration
	// Counts is the number of TrainJobs in each state, including states with no TrainJobs.
	Counts    map[TrainJobState]int
	TrainJobs []TrainJobReport
}

// MonitorTrainJobs lists the TrainJobs and reports their current state and progress.
func MonitorTrainJobs(ctx context.Context, c client.Client, now time.Time, progressDeadline time.Duration, opts ...client.ListOption) (*TrainJobsReport, error) {
	var trainJobs trainer.TrainJobList
	if err := c.List(ctx, &trainJobs, opts...); err != nil {
		return nil, fmt.Errorf("failed to list TrainJobs: %w", err)
	}
	return NewTrainJobsReport(trainJobs.Items, now, progressDeadline), nil
}

// NewTrainJobsReport reports the state and progress of every TrainJob, sorted by namespace and name.
func NewTrainJobsReport(trainJobs []trainer.TrainJob, now time.Time, progressDeadline time.Duration) *TrainJobsReport {
	report := &TrainJobsReport{
		GeneratedAt:      now,
		ProgressDeadline: progressDeadline,
		Counts:           make(map[TrainJobState]int, len(trainJobStates)),
	}
	for _, state := range trainJobStates {
		report.Counts[state] = 0
	}
	for i := range trainJobs {
		trainJob := &trainJobs[i]
		state := GetTrainJobState(trainJob, now, progressDeadline)
		report.Counts[state]++
		trainJobReport := TrainJobReport{
			Namespace: trainJob.Namespace,
			Name:      trainJob.Name,
			State:     state,
		}
		if trainerStatus := trainJob.Status.TrainerStatus; trainerStatus != nil {
			if trainerStatus.ProgressPercentage != nil {
				trainJobReport.ProgressPercentage = ptr.To(*trainerStatus.ProgressPercentage)
			}
			trainJobReport.LastUpdatedTime = trainerStatus.LastUpdatedTime
			trainJobReport.LastProgressTime = trainerStatus.LastProgressTime
		}
		report.TrainJobs = append(report.TrainJobs, trainJobReport)
	}
	slices.SortFunc(report.TrainJobs, func(a, b TrainJobReport) int {
		return cmp.Or(cmp.Compare(a.Namespace, b.Namespace), cmp.Compare(a.Name, b.Name))
	})
	return report
}

// GetTrainJobState returns the state of the TrainJob as reported by MonitorTrainJobs.
func GetTrainJobState(trainJob *trainer.TrainJob, now time.Time, progressDeadline time.Duration) TrainJobState {
	if failed := meta.FindStatusCondition(trainJob.Status.Conditions, trainer.TrainJobFailed); failed != nil && failed.Status == metav1.ConditionTrue {
		if failed.Reason == trainer.TrainJobProgressDeadlineExceededReason {
			return TrainJobStateKilled
		}
		return TrainJobStateFailed
	}
	switch {
	case meta.IsStatusConditionTrue(trainJob.Status.Conditions, trainer.TrainJobComplete):
		return TrainJobStateCompleted
	case ptr.Deref(trainJob.Spec.Suspend, false):
		return TrainJobStateSuspended
	case IsTrainJobStalled(trainJob, now, progressDeadline):
		return TrainJobStateStalled
	default:
		return TrainJobStateActive
	}
}

// String renders the report as the number of TrainJobs in each state, followed by a table of
// every TrainJob with its progress bar and the time since its last update and progress.
func (r *TrainJobsReport) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "TrainJobs report at %s (progress deadline: %s)\n", r.GeneratedAt.UTC().Format(time.RFC3339), r.ProgressDeadline)
	fmt.Fprintf(&b, "Total: %d", len(r.TrainJobs))
	for _, state := range trainJobStates {
		fmt.Fprintf(&b, "  %s: %d", state, r.Counts[state])
	}
	b.WriteString("\n\n")

	age := func(t metav1.Time) string {
		if t.IsZero() {
			return "-"
		}
		return duration.ShortHumanDuration(r.GeneratedAt.Sub(t.Time))
	}
	// Writes to the tabwriter can't fail, since it's backed by a strings.Builder.
	w := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(w, "NAMESPACE\tNAME\tSTATE\tPROGRESS\tLAST UPDATE\tLAST PROGRESS")
	for _, trainJob := range r.TrainJobs {
		_, _ = fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", trainJob.Namespace, trainJob.Name, trainJob.State,
			progressBar(trainJob.ProgressPercentage), age(trainJob.LastUpdatedTime), age(trainJob.LastProgressTime))
	}
	_ = w.Flush()
	return b.String()
}

// progressBar renders the progress percentage as a bar, e.g. "[██████████░░░░░░░░░░]  50%",
// or "-" when the progress is unknown.
func progressBar(progressPercentage *int32) string {
	if progressPercentage == nil {
		return "-"
	}
	percentage := min(max(*progressPercentage, 0), 100)
	filled := int(percentage) * progressBarWidth / 100
	return fmt.Sprintf("[%s%s] %3d%%", strings.Repeat("█", filled), strings.Repeat("░", progressBarWidth-filled), percentage)
}
