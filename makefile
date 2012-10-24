# makefile

# perform self-tests by running all the unit tests
# fail if any test fails
.PHONY: check
check:
	torch Estimator-test.lua
	torch EstimatorAvg-test.lua
	torch EstimatorKwavg-test.lua
	torch Nn-test.lua
	torch Smoother-test.lua
	torch SmootherAvg-test.lua
	torch SmootherKwavg-test.lua



