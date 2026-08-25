// Command cda demonstrates continuous chaotic DA on Rastrigin.
package main

import (
	"fmt"
	"log"

	"github.com/CWBudde/dragonfly"
)

func main() {
	config := dragonfly.NewChaoticConfig()
	config.ProblemSize = 10
	config.LowerBound = -5.12
	config.UpperBound = 5.12
	config.MaxIterations = 300
	config.ObjectiveFunc = dragonfly.Rastrigin
	seed := int64(42)
	config.Seed = &seed

	result, err := dragonfly.OptimizeChaotic(config)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("CDA best cost %.6g using %s chaos\n",
		result.GlobalBest.Cost, config.ChaosMap)
}
