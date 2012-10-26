# makefile

# perform self-tests by running all the unit tests
# fail if any test fails
.PHONY: check
check:
	torch NnwEstimator-test.lua
	torch NnwEstimatorAvg-test.lua
	torch NnwEstimatorKwavg-test.lua
	torch NnwEstimatorLlr-test.lua
	torch Nnw-test.lua
	torch NnwSmoother-test.lua
	torch NnwSmootherAvg-test.lua
	torch NnwSmootherKwavg-test.lua
	torch NnwSmootherLlr-test.lua



