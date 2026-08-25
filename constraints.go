// Constraint aggregation, Deb's feasibility rules and penalty ranking.

package dragonfly

import (
	"fmt"
	"math"
)

// ConstraintEvaluation describes the aggregate constraint state of a position.
// A zero violation is feasible; any positive violation measures how far the
// position sits outside the feasible region.
type ConstraintEvaluation struct {
	Violation float64
	Feasible  bool
}

// CandidateEvaluation carries the two numbers the ranking rules compare: the
// raw objective cost and the aggregate constraint violation.
//
// The cost stays raw on purpose. A penalized score is derived on demand by
// PenalizedCost, so a Result still reports the cost the caller's objective
// actually returned rather than a number the constraint policy invented.
type CandidateEvaluation struct {
	Cost                float64
	ConstraintViolation float64
}

// EvaluateConstraints evaluates and aggregates every configured constraint.
//
// Inequality constraints contribute max(0, g(x)); equality constraints
// contribute max(0, |h(x)| - tolerance). A nil config means the problem is
// unconstrained and is reported feasible without calling anything.
//
// A nil constraint function or a non-finite constraint value produces an
// infinite violation rather than an error: an unusable constraint has to lose
// every comparison, and infinity is the only violation that reliably does.
func EvaluateConstraints(position []float64, config *ConstraintConfig) ConstraintEvaluation {
	if config == nil {
		return ConstraintEvaluation{Feasible: true}
	}

	violation, ok := sumInequalityViolations(position, config.Inequalities)
	if !ok {
		return infeasibleEvaluation()
	}

	equalityViolation, ok := sumEqualityViolations(position, config.Equalities, effectiveEqualityTolerance(config))
	if !ok {
		return infeasibleEvaluation()
	}

	violation += equalityViolation
	if math.IsInf(violation, 1) || math.IsNaN(violation) {
		return infeasibleEvaluation()
	}

	return ConstraintEvaluation{Violation: violation, Feasible: violation == 0}
}

// infeasibleEvaluation is the verdict for a constraint that could not be
// evaluated: infinitely violated, and never feasible.
func infeasibleEvaluation() ConstraintEvaluation {
	return ConstraintEvaluation{Violation: math.Inf(1)}
}

// sumInequalityViolations accumulates max(0, g(x)) over the inequality
// constraints. The bool is false when a constraint was nil or returned a value
// the aggregation cannot use.
func sumInequalityViolations(position []float64, constraints []ConstraintFunction) (float64, bool) {
	violation := 0.0

	for _, constraint := range constraints {
		value, ok := constraintValue(position, constraint)
		if !ok {
			return 0, false
		}

		violation += max(0, value)
	}

	return violation, true
}

// sumEqualityViolations accumulates max(0, |h(x)| - tolerance) over the
// equality constraints, so an equality is satisfied exactly when it holds to
// within tolerance.
func sumEqualityViolations(
	position []float64,
	constraints []ConstraintFunction,
	tolerance float64,
) (float64, bool) {
	violation := 0.0

	for _, constraint := range constraints {
		value, ok := constraintValue(position, constraint)
		if !ok {
			return 0, false
		}

		violation += max(0, math.Abs(value)-tolerance)
	}

	return violation, true
}

// constraintValue calls one constraint and reports whether the result is usable.
func constraintValue(position []float64, constraint ConstraintFunction) (float64, bool) {
	if constraint == nil {
		return 0, false
	}

	value := constraint(position)
	if !isFinite(value) {
		return 0, false
	}

	return value, true
}

// IsFeasible reports whether an aggregate constraint violation is zero.
func IsFeasible(violation float64) bool {
	return violation == 0
}

// PenalizedCost folds an aggregate violation into a raw objective cost.
//
// PenaltyLinear adds factor*violation; PenaltyQuadratic adds
// factor*violation². An empty method defaults to quadratic, which is the usual
// choice: it leaves small violations almost free and makes large ones
// prohibitive, so the swarm can cross a thin infeasible ridge without settling
// on the far side of a thick one.
func PenalizedCost(cost, violation, factor float64, method PenaltyMethod) float64 {
	if method == PenaltyLinear {
		return cost + factor*violation
	}

	return cost + factor*violation*violation
}

// BetterConstrainedCandidate reports whether candidate is preferred over
// incumbent under config. A nil config falls back to ordinary objective
// minimization.
//
// The default policy is Deb's three feasibility rules:
//
//  1. a feasible candidate beats an infeasible one;
//  2. between two infeasible candidates, the smaller violation wins;
//  3. between two feasible candidates, the smaller cost wins.
//
// Deb's rules need no penalty factor to tune, which is why they are the
// default. ConstraintHandlingPenalty instead ranks by penalized score and only
// consults the feasibility rules to break an exact tie, so that two candidates
// with the same score are still ordered by feasibility rather than arbitrarily.
func BetterConstrainedCandidate(candidate, incumbent CandidateEvaluation, config *ConstraintConfig) bool {
	if config == nil {
		return candidate.Cost < incumbent.Cost
	}

	if effectiveConstraintHandling(config) == ConstraintHandlingPenalty {
		candidateScore := penalizedScore(candidate, config)

		incumbentScore := penalizedScore(incumbent, config)
		if candidateScore != incumbentScore {
			return candidateScore < incumbentScore
		}
	}

	return betterByFeasibilityRules(candidate, incumbent)
}

// penalizedScore applies config's penalty policy to one candidate.
func penalizedScore(candidate CandidateEvaluation, config *ConstraintConfig) float64 {
	return PenalizedCost(
		candidate.Cost,
		candidate.ConstraintViolation,
		effectivePenaltyFactor(config),
		effectivePenaltyMethod(config),
	)
}

// betterByFeasibilityRules applies Deb's three rules, in order.
func betterByFeasibilityRules(candidate, incumbent CandidateEvaluation) bool {
	candidateFeasible := IsFeasible(candidate.ConstraintViolation)

	incumbentFeasible := IsFeasible(incumbent.ConstraintViolation)
	if candidateFeasible != incumbentFeasible {
		return candidateFeasible
	}

	if !candidateFeasible && candidate.ConstraintViolation != incumbent.ConstraintViolation {
		return candidate.ConstraintViolation < incumbent.ConstraintViolation
	}

	return candidate.Cost < incumbent.Cost
}

// effectiveConstraintHandling reports the ranking policy a run will use. An
// unset field resolves to Deb's feasibility rules, the policy that needs no
// tuning. The value is never written back to the config.
func effectiveConstraintHandling(config *ConstraintConfig) ConstraintHandlingMethod {
	if config == nil {
		return ConstraintHandlingFeasibility
	}

	if config.Handling == ConstraintHandlingPenalty {
		return ConstraintHandlingPenalty
	}

	return ConstraintHandlingFeasibility
}

// effectivePenaltyMethod reports the penalty shape a run will use, resolving an
// unset field to PenaltyQuadratic.
func effectivePenaltyMethod(config *ConstraintConfig) PenaltyMethod {
	if config != nil && config.PenaltyMethod == PenaltyLinear {
		return PenaltyLinear
	}

	return PenaltyQuadratic
}

// effectivePenaltyFactor reports the penalty factor a run will use. A nil
// config or a value validateConstraintBlock would have rejected resolves to
// zero, which reduces the penalized score to the raw cost instead of poisoning
// every comparison with a NaN.
func effectivePenaltyFactor(config *ConstraintConfig) float64 {
	if config == nil || !isFinite(config.PenaltyFactor) || config.PenaltyFactor < 0 {
		return 0
	}

	return config.PenaltyFactor
}

// effectiveEqualityTolerance reports the tolerance an equality constraint is
// judged by. An unset field means exact equality; a value
// validateConstraintBlock would have rejected also resolves to zero.
func effectiveEqualityTolerance(config *ConstraintConfig) float64 {
	if config == nil || !isFinite(config.EqualityTolerance) || config.EqualityTolerance < 0 {
		return 0
	}

	return config.EqualityTolerance
}

// validateConstraintBlock rejects an optional constraint block the optimizer
// cannot run. A nil block means the problem is unconstrained and is valid.
//
// validateConfig is expected to wrap the returned error as
// "invalid constraint config: %w".
func validateConstraintBlock(config *ConstraintConfig) error {
	if config == nil {
		return nil
	}

	if !isFinite(config.EqualityTolerance) || config.EqualityTolerance < 0 {
		return fmt.Errorf("equality tolerance must be finite and non-negative, got %v",
			config.EqualityTolerance)
	}

	for i, constraint := range config.Inequalities {
		if constraint == nil {
			return fmt.Errorf("inequality constraint %d is nil", i)
		}
	}

	for i, constraint := range config.Equalities {
		if constraint == nil {
			return fmt.Errorf("equality constraint %d is nil", i)
		}
	}

	if !isFinite(config.PenaltyFactor) || config.PenaltyFactor < 0 {
		return fmt.Errorf("penalty factor must be finite and non-negative, got %v",
			config.PenaltyFactor)
	}

	switch config.PenaltyMethod {
	case "", PenaltyLinear, PenaltyQuadratic:
	default:
		return fmt.Errorf("unknown penalty method %q", config.PenaltyMethod)
	}

	return validateConstraintHandling(config)
}

// validateConstraintHandling checks the ranking policy and the one field it
// makes mandatory: a penalty factor of zero turns the penalty policy into plain
// cost minimization, which silently ignores the constraints the caller wrote.
func validateConstraintHandling(config *ConstraintConfig) error {
	switch config.Handling {
	case "", ConstraintHandlingFeasibility:
		return nil
	case ConstraintHandlingPenalty:
		if config.PenaltyFactor == 0 {
			return fmt.Errorf("penalty factor must be finite and positive, got %v", config.PenaltyFactor)
		}

		return nil
	default:
		return fmt.Errorf("unknown constraint handling method %q", config.Handling)
	}
}

// constraintEvaluator pairs the objective with the constraint policy, so the
// main loop has a single place to ask "what does this position cost" and "is
// this candidate better", whether or not the problem is constrained.
type constraintEvaluator struct {
	objective   ObjectiveFunction
	constraints *ConstraintConfig
}

// newConstraintEvaluator builds an evaluator. A nil constraints block yields an
// evaluator that behaves exactly like unconstrained cost minimization.
func newConstraintEvaluator(objective ObjectiveFunction, constraints *ConstraintConfig) *constraintEvaluator {
	return &constraintEvaluator{objective: objective, constraints: constraints}
}

// evaluate scores one position. Constraints are evaluated first so that a
// position is measured against the same policy regardless of what the objective
// returns. When sanitize is set, an unusable cost is mapped to +Inf.
func (evaluator *constraintEvaluator) evaluate(position []float64, sanitize bool) CandidateEvaluation {
	constraint := EvaluateConstraints(position, evaluator.constraints)

	cost := evaluator.objective(position)
	if sanitize {
		cost = sanitizeCost(cost)
	}

	return CandidateEvaluation{
		Cost:                cost,
		ConstraintViolation: constraint.Violation,
	}
}

// evaluateDragonfly scores a dragonfly's position in place.
func (evaluator *constraintEvaluator) evaluateDragonfly(dragonfly *Dragonfly) {
	evaluation := evaluator.evaluate(dragonfly.Position, true)
	dragonfly.Cost = evaluation.Cost
	dragonfly.ConstraintViolation = evaluation.ConstraintViolation
}

// better reports whether candidate is preferred over incumbent under the
// evaluator's policy.
func (evaluator *constraintEvaluator) better(candidate, incumbent CandidateEvaluation) bool {
	return BetterConstrainedCandidate(candidate, incumbent, evaluator.constraints)
}

// betterDragonfly compares two swarm members.
func (evaluator *constraintEvaluator) betterDragonfly(candidate, incumbent *Dragonfly) bool {
	return evaluator.better(evaluationFromDragonfly(candidate), evaluationFromDragonfly(incumbent))
}

// betterDragonflyThanBest compares a swarm member against the incumbent food
// source or enemy.
func (evaluator *constraintEvaluator) betterDragonflyThanBest(candidate *Dragonfly, incumbent Best) bool {
	return evaluator.better(evaluationFromDragonfly(candidate), evaluationFromBest(incumbent))
}

// betterBest compares two recorded bests.
func (evaluator *constraintEvaluator) betterBest(candidate, incumbent Best) bool {
	return evaluator.better(evaluationFromBest(candidate), evaluationFromBest(incumbent))
}

// feasible reports whether a scored position satisfies every constraint.
func (evaluator *constraintEvaluator) feasible(candidate CandidateEvaluation) bool {
	return IsFeasible(candidate.ConstraintViolation)
}

// evaluationFromDragonfly views a swarm member as a ranking candidate.
func evaluationFromDragonfly(dragonfly *Dragonfly) CandidateEvaluation {
	return CandidateEvaluation{Cost: dragonfly.Cost, ConstraintViolation: dragonfly.ConstraintViolation}
}

// evaluationFromBest views a recorded best as a ranking candidate.
func evaluationFromBest(best Best) CandidateEvaluation {
	return CandidateEvaluation{Cost: best.Cost, ConstraintViolation: best.ConstraintViolation}
}

// bestFromDragonfly snapshots a swarm member as a Best, copying the position so
// the snapshot does not alias the swarm.
func bestFromDragonfly(dragonfly *Dragonfly) Best {
	return Best{
		Position:            copyVec(dragonfly.Position),
		Cost:                dragonfly.Cost,
		ConstraintViolation: dragonfly.ConstraintViolation,
	}
}

// copyDragonflyToBest overwrites destination in place, reusing its position
// slice. destination.Position must already have the right length.
func copyDragonflyToBest(destination *Best, source *Dragonfly) {
	destination.Cost = source.Cost
	destination.ConstraintViolation = source.ConstraintViolation
	copy(destination.Position, source.Position)
}
