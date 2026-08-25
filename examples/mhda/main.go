// Command mhda demonstrates the memory-based DA/PSO hybrid.
package main

import (
	"fmt"
	"log"

	"github.com/CWBudde/dragonfly"
)

func main() {
	config := dragonfly.NewMemoryHybridConfig()
	config.ObjectiveFunc = dragonfly.Sphere
	config.ProblemSize = 10
	config.LowerBound = -100
	config.UpperBound = 100
	config.MaxIterations = 300
	seed := int64(42)
	config.Seed = &seed

	result, err := dragonfly.OptimizeMemoryHybrid(config)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("MHDA best %.6g after %d evaluations\n",
		result.GlobalBest.Cost, result.FuncEvalCount)
}
