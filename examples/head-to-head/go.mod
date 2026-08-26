module github.com/CWBudde/dragonfly/examples/head-to-head

go 1.23.3

require (
	github.com/CWBudde/dragonfly v0.0.0
	github.com/CWBudde/go-cma-es v0.0.0
	github.com/cwbudde/mayfly v0.0.0
)

require github.com/cwbudde/qmc v0.2.0 // indirect

replace github.com/CWBudde/dragonfly => ../..

replace github.com/CWBudde/go-cma-es => ../../../go-cma-es

replace github.com/cwbudde/mayfly => ../../../Mayfly
