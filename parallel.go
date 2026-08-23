// Deterministic parallel evaluation: the worker pool that scores a swarm, and
// the index-stable reduction that folds the scores back into the food source
// and the enemy.
//
// The division of labor here is the whole point of the parallel path. Workers
// call the objective function and write one CandidateEvaluation each; they
// never draw a random number and they never touch the swarm. Everything that
// consumes the RNG -- the weight schedules, the neighborhood scan, the step
// update, the boundary repair -- has already happened on the calling goroutine
// by the time a single worker starts, which is what makes a seeded run
// bit-identical with EnableParallel on or off.

package dragonfly

import (
	"context"
	"sync"
)

// rankFunc reports whether candidate is preferred over incumbent.
//
// It exists so one reduction can find both extremes of a batch: the food
// source uses the evaluator's ordering, the enemy uses the same ordering read
// backwards. Writing the enemy as "the best under a reversed comparison"
// rather than as a second, mirrored implementation is what guarantees the two
// stay in step when the constraint policy changes.
type rankFunc func(candidate, incumbent CandidateEvaluation) bool

// evaluationPool scores a swarm with bounded concurrency.
//
// The scratch buffer is reused across iterations. It is what keeps the pool a
// pool rather than a bare function call: a run allocates one slice of
// evaluations and reuses it for every batch, and the swarm itself is written
// only after the whole batch has succeeded.
type evaluationPool struct {
	evaluator *constraintEvaluator
	// scores[i] holds the evaluation of swarm[i] for the batch in flight. Each
	// worker writes exactly one element and reads none, so no synchronization
	// beyond the pool's WaitGroup join is needed.
	scores     []CandidateEvaluation
	maxWorkers int
}

// newEvaluationPool builds a pool for a run of the given swarm size. A
// non-positive capacity is legal; the buffer grows on first use.
func newEvaluationPool(evaluator *constraintEvaluator, maxWorkers, capacity int) *evaluationPool {
	if capacity < 0 {
		capacity = 0
	}

	return &evaluationPool{
		evaluator:  evaluator,
		scores:     make([]CandidateEvaluation, 0, capacity),
		maxWorkers: maxWorkers,
	}
}

// evaluatedBatch is the outcome of one parallel evaluation pass: the raw
// scores in swarm order, plus the two reductions the food source and the enemy
// are merged from. It exists so the pass can hand back all three without a
// four-value return.
type evaluatedBatch struct {
	best   *batchBest
	worst  *batchBest
	scores []CandidateEvaluation
}

// evaluate scores every dragonfly in swarm and reduces the batch to its best
// and worst member, without writing anything back to the swarm.
//
// Nothing is committed here on purpose. A canceled batch returns an error and
// leaves the swarm exactly as it found it, so a caller cannot mistake a
// half-evaluated population for a complete one -- the costs in the swarm are
// always all from the same iteration or all from the previous one.
//
// The returned scores alias the pool's scratch buffer and stay valid only
// until the next call.
func (pool *evaluationPool) evaluate(ctx context.Context, swarm []Dragonfly) (evaluatedBatch, error) {
	if cap(pool.scores) < len(swarm) {
		pool.scores = make([]CandidateEvaluation, len(swarm))
	}

	batch := evaluatedBatch{
		best:   newBatchBest(pool.evaluator),
		worst:  newBatchWorst(pool.evaluator),
		scores: pool.scores[:len(swarm)],
	}

	workErr := parallelFor(ctx, len(swarm), pool.maxWorkers, func(i int) {
		// Position is read-only for the duration of the batch: the step update
		// that produced it ran to completion on the calling goroutine before
		// this pool was handed the swarm.
		evaluation := pool.evaluator.evaluate(swarm[i].Position, true)
		batch.scores[i] = evaluation

		batch.best.consider(i, swarm[i].Position, evaluation)
		batch.worst.consider(i, swarm[i].Position, evaluation)
	})
	if workErr != nil {
		return evaluatedBatch{}, workErr
	}

	return batch, nil
}

// batchBest is the extreme member of one evaluated batch, selected by stable
// index.
//
// Ties are the reason this type is not a plain running minimum. Workers see the
// swarm in an order that depends on goroutine scheduling, so a reduction that
// keeps the first candidate it happens to see would resolve two equally good
// dragonflies differently from run to run. Keeping the LOWEST INDEX among equal
// candidates instead reproduces exactly what the sequential loop does -- it
// scans in index order and replaces only on a strict improvement, so the
// earliest of several equal candidates wins -- and that is what makes a
// parallel run bit-identical to a sequential one rather than merely as good.
type batchBest struct {
	rank rankFunc
	best Best
	mu   sync.Mutex
	// index is the swarm index of the incumbent, or -1 while the batch is still
	// empty. It is the tie-breaker, and it doubles as the "anything considered
	// yet" flag so no sentinel cost has to be invented for either direction.
	index int
}

// newBatchBest reduces a batch to its best member under the evaluator's
// ordering: the candidate the food source is taken from.
func newBatchBest(evaluator *constraintEvaluator) *batchBest {
	return &batchBest{rank: evaluator.better, index: -1}
}

// newBatchWorst reduces a batch to its worst member: the candidate the enemy is
// taken from. It is the same reduction with the comparison reversed, which is
// how the sequential loop finds the enemy as well.
func newBatchWorst(evaluator *constraintEvaluator) *batchBest {
	reversed := func(candidate, incumbent CandidateEvaluation) bool {
		return evaluator.better(incumbent, candidate)
	}

	return &batchBest{rank: reversed, index: -1}
}

// consider offers one scored dragonfly to the reduction. It is safe to call
// from several goroutines at once, and its result does not depend on the order
// in which they call it.
func (batch *batchBest) consider(index int, position []float64, evaluation CandidateEvaluation) {
	batch.mu.Lock()
	defer batch.mu.Unlock()

	if !batch.accepts(index, evaluation) {
		return
	}

	batch.best.Cost = evaluation.Cost
	batch.best.ConstraintViolation = evaluation.ConstraintViolation
	batch.best.Position = append(batch.best.Position[:0], position...)
	batch.index = index
}

// accepts reports whether a candidate displaces the incumbent. The caller holds
// the lock.
//
// A strictly better candidate always wins. An equally good one wins only when
// it sits earlier in the swarm, which is the stable-index rule the type exists
// for.
func (batch *batchBest) accepts(index int, evaluation CandidateEvaluation) bool {
	if batch.index < 0 {
		return true
	}

	incumbent := evaluationFromBest(batch.best)

	if batch.rank(evaluation, incumbent) {
		return true
	}

	if batch.rank(incumbent, evaluation) {
		return false
	}

	return index < batch.index
}

// mergeBest folds a finished batch into a run-level incumbent -- the food
// source or the enemy -- replacing it only on a strict improvement.
//
// That "only on a strict improvement" is what makes the two-stage reduction
// agree with the sequential loop's single pass. The sequential loop compares
// every candidate against a running incumbent that starts at the previous
// value; reducing the batch on its own first and merging afterwards picks the
// same element, because an incumbent that ties the batch's extreme keeps its
// place in both formulations.
//
// destination.Position must already have the right length; it is overwritten in
// place rather than reallocated, matching copyDragonflyToBest.
func mergeBest(destination *Best, batch *batchBest) {
	batch.mu.Lock()
	defer batch.mu.Unlock()

	if batch.index < 0 {
		return
	}

	if !batch.rank(evaluationFromBest(batch.best), evaluationFromBest(*destination)) {
		return
	}

	destination.Cost = batch.best.Cost
	destination.ConstraintViolation = batch.best.ConstraintViolation
	copy(destination.Position, batch.best.Position)
}
