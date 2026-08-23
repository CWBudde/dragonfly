// Command constrained solves a constrained problem under Deb's feasibility
// rules and shows what the reported solution does and does not contain.
//
// The problem is a two-variable least-distance problem with a known answer:
//
//	minimise   f(x) = (x0 - 3)^2 + (x1 - 2)^2
//	subject to g1(x) = x0 + x1 - 3 <= 0
//	           g2(x) = -x0         <= 0
//	           g3(x) = -x1         <= 0
//
// The unconstrained minimum is (3, 2) with f = 0, and it is infeasible: it
// violates g1 by 2. The constrained minimum is the projection of (3, 2) onto
// the line x0 + x1 = 3, namely (2, 1) with f = 2. Knowing both numbers is the
// point -- an optimizer that quietly ignored the constraint would report a cost
// below 2, and one that folded a penalty into the cost would report a cost
// above f(x) at its own solution.
//
// Two things are demonstrated:
//
//  1. the reported solution is feasible -- Result.GlobalBest.ConstraintViolation
//     is exactly zero, and each constraint is re-checked here rather than taken
//     on trust;
//  2. Result.GlobalBest.Cost is the raw objective. It is recomputed from the
//     reported position and compared bit for bit. A penalized score is
//     something the caller derives on demand with PenalizedCost; it is never
//     what Result carries.
//
// A second run keeps the same constraints but switches the ranking policy to
// ConstraintHandlingPenalty with a deliberately weak factor, so the swarm
// settles on a slightly infeasible point that a stronger factor would have
// rejected. That is the case where the raw-versus-penalized distinction is
// visible rather than merely stated: the reported cost drops below the true
// constrained optimum, because it is f(x) at a point outside the feasible set.
//
// Run it with:
//
//	go run .
package main

import (
	"fmt"
	"math"
	"math/rand"

	dragonfly "github.com/CWBudde/Dragonfly"
)

const (
	problemSize   = 2
	bound         = 5.0
	maxIterations = 400
	// optimizerSeed pins the random stream so both runs below are reproducible.
	optimizerSeed = 11
	// penaltyFactor is the second run's penalty weight, and it is deliberately
	// far too small. With a quadratic penalty the score f(x) + factor*v^2 is
	// minimised at a violation of v = 2/(1 + 2*factor), which is 2/3 here, so
	// the policy itself prefers an infeasible point. Deb's feasibility rules
	// need no factor to tune, which is why they are the default.
	penaltyFactor = 1.0
)

// objective is the raw cost: the squared distance to the unconstrained
// optimum (3, 2).
func objective(x []float64) float64 {
	return (x[0]-3)*(x[0]-3) + (x[1]-2)*(x[1]-2)
}

// budget is g1: the two variables together may not exceed three.
func budget(x []float64) float64 {
	return x[0] + x[1] - 3
}

// nonNegativeFirst and nonNegativeSecond keep both variables non-negative. They
// are inactive at the optimum and are there so the example has more than one
// constraint to aggregate.
func nonNegativeFirst(x []float64) float64  { return -x[0] }
func nonNegativeSecond(x []float64) float64 { return -x[1] }

func main() {
	fmt.Println("Constrained Dragonfly run -- Deb's feasibility rules")
	fmt.Println()
	fmt.Println("minimise (x0-3)^2 + (x1-2)^2   subject to x0 + x1 <= 3, x0 >= 0, x1 >= 0")
	fmt.Println("unconstrained optimum: (3, 2), f = 0, infeasible (violates the budget by 2)")
	fmt.Println("constrained optimum  : (2, 1), f = 2")
	fmt.Println()

	inequalityRun()
	fmt.Println()
	penaltyRun()
}

// inequalityRun is the main case: the run converges into the feasible region
// and the reported violation is exactly zero.
func inequalityRun() {
	config := baseConfig()
	config.Constraints = &dragonfly.ConstraintConfig{
		Handling: dragonfly.ConstraintHandlingFeasibility,
		Inequalities: []dragonfly.ConstraintFunction{
			budget, nonNegativeFirst, nonNegativeSecond,
		},
	}

	result, err := dragonfly.Optimize(config)
	if err != nil {
		fmt.Println("optimization failed:", err)

		return
	}

	best := result.GlobalBest

	fmt.Println("== inequality-constrained run ==")
	fmt.Printf("solution         : (%.6f, %.6f)\n", best.Position[0], best.Position[1])
	fmt.Printf("reported cost    : %.6f      (known constrained optimum: 2)\n", best.Cost)
	fmt.Printf("violation        : %v\n", best.ConstraintViolation)
	fmt.Printf("feasible         : %v\n", dragonfly.IsFeasible(best.ConstraintViolation))
	fmt.Println()
	fmt.Println("constraint by constraint, re-checked here rather than taken on trust:")
	fmt.Printf("  g1 = x0 + x1 - 3 = %+.3e  (satisfied when <= 0: %v)\n",
		budget(best.Position), budget(best.Position) <= 0)
	fmt.Printf("  g2 = -x0         = %+.3e  (satisfied when <= 0: %v)\n",
		nonNegativeFirst(best.Position), nonNegativeFirst(best.Position) <= 0)
	fmt.Printf("  g3 = -x1         = %+.3e  (satisfied when <= 0: %v)\n",
		nonNegativeSecond(best.Position), nonNegativeSecond(best.Position) <= 0)
	fmt.Println()
	reportRawCost(best)
}

// penaltyRun is the contrasting case: the same problem ranked by penalized
// score with a factor far too small to enforce the budget. The swarm settles
// just outside the feasible region, and the reported cost -- still the raw
// objective -- lands below the true constrained optimum of 2 because it is
// measured at a point that is not allowed.
func penaltyRun() {
	config := baseConfig()
	config.Constraints = &dragonfly.ConstraintConfig{
		Handling:      dragonfly.ConstraintHandlingPenalty,
		PenaltyMethod: dragonfly.PenaltyQuadratic,
		PenaltyFactor: penaltyFactor,
		Inequalities: []dragonfly.ConstraintFunction{
			budget, nonNegativeFirst, nonNegativeSecond,
		},
	}

	result, err := dragonfly.Optimize(config)
	if err != nil {
		fmt.Println("optimization failed:", err)

		return
	}

	best := result.GlobalBest
	score := dragonfly.PenalizedCost(
		best.Cost, best.ConstraintViolation, penaltyFactor, dragonfly.PenaltyQuadratic)

	fmt.Printf("== penalty-ranked run, quadratic, factor %g ==\n", penaltyFactor)
	fmt.Printf("solution         : (%.6f, %.6f)\n", best.Position[0], best.Position[1])
	fmt.Printf("reported cost    : %.6f      (below the constrained optimum of 2 -- see below)\n", best.Cost)
	fmt.Printf("violation        : %.6f      (g1 = %+.6f, so the budget is overspent)\n",
		best.ConstraintViolation, budget(best.Position))
	fmt.Printf("feasible         : %v\n", dragonfly.IsFeasible(best.ConstraintViolation))
	fmt.Println()
	reportRawCost(best)
	fmt.Println()
	fmt.Printf("penalized score  : %.6f = cost + %g * violation^2\n", score, penaltyFactor)
	fmt.Printf("the score exceeds the reported cost by %.6f, and it is the score -- never\n",
		math.Abs(score-best.Cost))
	fmt.Println("the reported cost -- that this policy ranked candidates by. Result carries")
	fmt.Println("the number the objective function returned; PenalizedCost derives the other")
	fmt.Println("on demand. Reading the cost alone here would understate the answer: it is")
	fmt.Println("f(x) at a point the constraints forbid, which is why the violation is")
	fmt.Println("reported next to it and why the default policy is Deb's rules.")
}

// reportRawCost recomputes the objective at the reported position and compares
// it with the reported cost. Equality is the proof that no penalty was folded
// in: a penalized score would sit above f(x) whenever the violation is nonzero.
func reportRawCost(best dragonfly.Best) {
	recomputed := objective(best.Position)

	fmt.Printf("objective recomputed at the reported position: %.12g\n", recomputed)
	fmt.Printf("Result.GlobalBest.Cost                      : %.12g\n", best.Cost)
	fmt.Printf("identical to the last bit: %v\n",
		math.Float64bits(recomputed) == math.Float64bits(best.Cost))
}

// baseConfig is the configuration both runs share.
//
// BoundaryMethod is clamp rather than the paper's wrap. Wrapping teleports a
// dragonfly that leaves the box to the opposite bound, which is good
// exploration on an unconstrained problem and actively unhelpful next to a
// constraint boundary that runs along one -- a candidate that has just crawled
// up to the edge of the feasible region reappears on the far side of the search
// space. Config.BoundaryMethod exists for exactly this trade.
func baseConfig() *dragonfly.Config {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = objective
	config.ProblemSize = problemSize
	config.LowerBound = -bound
	config.UpperBound = bound
	config.MaxIterations = maxIterations
	config.BoundaryMethod = dragonfly.BoundaryClamp
	config.Rand = rand.New(rand.NewSource(optimizerSeed))

	return config
}
