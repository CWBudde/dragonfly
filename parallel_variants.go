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

// evaluateParallelBinary scores a binary swarm through the pool and updates the
// food source and the enemy, returning the number of objective calls it made.
//
// It is the parallel counterpart of the binary loop's sequential evaluation and
// the BDA analog of evaluateParallelStep. The reduction is deliberately the
// same one rather than a mirrored copy: BDA differs from DA only in how a
// position is produced, never in how one is scored, so a second implementation
// here could only drift from the first. The objective keeps the ordinary
// ObjectiveFunction signature and is handed a 0/1 vector, and the food source
// and the enemy are ranked by the same evaluator.
//
// What the binary variant does need is the same guarantee about where the
// randomness lives. Every draw one iteration makes -- the weight schedules, the
// step update and, unique to BDA, the per-component bit-flip test in flipBits
// -- has already happened on the calling goroutine before this function is
// called. Workers read a finished 0/1 position and call the objective; a single
// draw moved in here would make the flip decisions depend on the interleaving
// of the workers instead of on the seed.
func evaluateParallelBinary(ctx context.Context, state *runState, pool *evaluationPool) (int, error) {
	return evaluateParallelStep(ctx, state, pool)
}
