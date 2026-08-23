// Early-termination bookkeeping for the Dragonfly Algorithm.

package dragonfly

// convergenceTracker watches the swarm-level incumbent between iterations and
// decides whether the run may stop before the iteration cap is reached.
//
// It owns the stagnation counter, so it must be fed exactly once per
// iteration, in order, with the incumbent as it stands after that iteration's
// evaluations.
type convergenceTracker struct {
	config             *ConvergenceConfig
	evaluator          *constraintEvaluator
	referenceBest      CandidateEvaluation
	stagnantIterations int
}

// newConvergenceTracker creates a tracker seeded with the incumbent that the
// initialization phase produced. The optional evaluator supplies the
// constraint policy used to decide what counts as an improvement; omitting it
// falls back to plain objective minimization.
func newConvergenceTracker(
	config *ConvergenceConfig,
	initialBest Best,
	evaluators ...*constraintEvaluator,
) *convergenceTracker {
	evaluator := newConstraintEvaluator(nil, nil)
	if len(evaluators) > 0 {
		evaluator = evaluators[0]
	}

	return &convergenceTracker{
		config:             config,
		evaluator:          evaluator,
		referenceBest:      evaluationFromBest(initialBest),
		stagnantIterations: 0,
	}
}

// observe records the incumbent for a completed iteration and reports whether
// the run should stop. The returned reason is empty when it should not.
//
// Both stopping criteria are gated by MinIterations, but the stagnation
// counter is maintained from the first observation regardless, so a run that
// stalls during the warm-up period stops as soon as the gate opens.
func (tracker *convergenceTracker) observe(iteration int, best Best) (TerminationReason, bool) {
	if tracker.config == nil {
		return "", false
	}

	bestEvaluation := evaluationFromBest(best)

	if tracker.significantlyImproved(bestEvaluation) {
		tracker.referenceBest = bestEvaluation
		tracker.stagnantIterations = 0
	} else {
		tracker.stagnantIterations++
	}

	minimumIterations := max(tracker.config.MinIterations, 1)
	if iteration < minimumIterations {
		return "", false
	}

	// A target cost is only meaningful for a feasible incumbent: an infeasible
	// position can undercut the target simply by ignoring the constraints.
	if tracker.config.TargetCost != nil && IsFeasible(bestEvaluation.ConstraintViolation) &&
		bestEvaluation.Cost <= *tracker.config.TargetCost {
		return TerminationTargetCost, true
	}

	if tracker.config.StagnationIterations > 0 &&
		tracker.stagnantIterations >= tracker.config.StagnationIterations {
		return TerminationStagnation, true
	}

	return "", false
}

// significantlyImproved reports whether candidate beats the tracked reference
// by more than MinImprovement. The margin is measured on whichever quantity
// the active constraint policy ranks by: the penalized score under penalty
// handling, the raw cost between two feasible candidates, and the aggregate
// violation between two infeasible ones. Crossing into feasibility always
// counts as an improvement, whatever the margin.
func (tracker *convergenceTracker) significantlyImproved(candidate CandidateEvaluation) bool {
	if !tracker.evaluator.better(candidate, tracker.referenceBest) {
		return false
	}

	if tracker.usesPenaltyHandling() {
		return tracker.penaltyScore(tracker.referenceBest)-tracker.penaltyScore(candidate) >
			tracker.config.MinImprovement
	}

	referenceFeasible := IsFeasible(tracker.referenceBest.ConstraintViolation)

	candidateFeasible := IsFeasible(candidate.ConstraintViolation)
	if referenceFeasible != candidateFeasible {
		return candidateFeasible
	}

	if candidateFeasible {
		return tracker.referenceBest.Cost-candidate.Cost > tracker.config.MinImprovement
	}

	return tracker.referenceBest.ConstraintViolation-candidate.ConstraintViolation >
		tracker.config.MinImprovement
}

// usesPenaltyHandling reports whether the tracked constraint policy ranks
// candidates by penalized cost.
func (tracker *convergenceTracker) usesPenaltyHandling() bool {
	return tracker.evaluator.constraints != nil &&
		tracker.evaluator.constraints.Handling == ConstraintHandlingPenalty
}

// penaltyScore folds an evaluation's constraint violation into its cost using
// the tracked penalty configuration. It is only meaningful under penalty
// handling, which usesPenaltyHandling establishes.
func (tracker *convergenceTracker) penaltyScore(evaluation CandidateEvaluation) float64 {
	config := tracker.evaluator.constraints

	return PenalizedCost(
		evaluation.Cost, evaluation.ConstraintViolation,
		config.PenaltyFactor, config.PenaltyMethod,
	)
}
