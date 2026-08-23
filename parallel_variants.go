// The parallel counterpart of runState.evaluateSwarm.

package dragonfly

import "context"

// evaluateParallelStep scores the swarm through the pool and updates the food
// source and the enemy, returning the number of objective calls it made.
//
// It is the mirror image of runState.evaluateSwarm and produces exactly the
// same state: the same costs, the same food, the same enemy, and the same
// evaluation count. The two differ only in where the objective calls happen.
//
// The commit is deliberately a second, sequential pass. Writing the scores back
// only after the whole batch succeeded is what makes a canceled iteration leave
// no trace: the swarm carries either every cost from this iteration or every
// cost from the previous one, never a mixture.
func evaluateParallelStep(ctx context.Context, state *runState, pool *evaluationPool) (int, error) {
	batch, err := pool.evaluate(ctx, state.swarm)
	if err != nil {
		return 0, err
	}

	for i := range state.swarm {
		fly := &state.swarm[i]
		fly.Cost = batch.scores[i].Cost
		fly.ConstraintViolation = batch.scores[i].ConstraintViolation
	}

	mergeBest(&state.food, batch.best)
	mergeBest(&state.enemy, batch.worst)

	return len(state.swarm), nil
}
