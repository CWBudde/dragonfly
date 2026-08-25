// Command qgda demonstrates quantum-behaved Gaussian mutational DA.
package main

import (
	"fmt"
	"log"

	"github.com/CWBudde/dragonfly"
)

func main() {
	config := dragonfly.NewQuantumConfig()
	config.ObjectiveFunc = dragonfly.Rastrigin
	config.ProblemSize = 10
	config.LowerBound = -5.12
	config.UpperBound = 5.12
	config.MaxIterations = 300
	seed := int64(42)
	config.Seed = &seed

	result, err := dragonfly.OptimizeQuantum(config)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("QGDA best %.6g after %d evaluations\n",
		result.GlobalBest.Cost, result.FuncEvalCount)
}
