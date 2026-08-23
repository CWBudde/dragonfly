// The bounded worker-pool primitive every parallel phase of a run is built on.

package dragonfly

import (
	"context"
	"sync"
	"sync/atomic"
)

// parallelFor runs work(0) .. work(count-1) with at most maxWorkers goroutines
// and returns once every invocation that started has finished.
//
// Indices are handed out by an atomic counter, so the order in which they run
// is not defined and must not matter. Each invocation must own its writes and
// read only data that is immutable for the duration of the call. In particular
// it must not draw a random number: every RNG draw in this package happens on
// the calling goroutine, and a draw made here would depend on the interleaving
// of the workers rather than on the seed.
//
// A canceled context stops the remaining indices from starting and is reported
// as the returned error. Work already in flight is allowed to finish, so a
// caller that returns on an error still knows no worker is touching its data.
// The error is ctx.Err() even when every index completed, because a run that
// was canceled must not be mistaken for one that ran to completion.
func parallelFor(ctx context.Context, count, maxWorkers int, work func(int)) error {
	if count <= 0 {
		return ctx.Err()
	}

	// A non-positive worker count would spawn no goroutines at all and silently
	// skip the work rather than doing it slowly.
	if maxWorkers < 1 {
		maxWorkers = 1
	}

	workerCount := min(count, maxWorkers)

	var next atomic.Int64

	var workers sync.WaitGroup

	workers.Add(workerCount)

	for range workerCount {
		go func() {
			defer workers.Done()

			for {
				if ctx.Err() != nil {
					return
				}

				index := int(next.Add(1) - 1)
				if index >= count {
					return
				}

				work(index)
			}
		}()
	}

	workers.Wait()

	return ctx.Err()
}
