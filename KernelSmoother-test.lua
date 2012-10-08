-- KernelSmoother-test.lua
-- unit test

require 'all'

test = {}
tester = Tester()

function test.euclideanDistances()
   local v = makeVerbose(false, 'test.euclideanDistance')
   xs = torch.Tensor(2,3):fill(0)
   xs[2][2] = 0.2
   v('xs', xs)
   local ys = torch.Tensor(2):fill(1)
   local kmax = 1
   local kernelSmoother = KernelSmoother()

   -- query not in xs
   local query = torch.Tensor(3):fill(0)
   v('query', query)
   local distances = kernelSmoother:euclideanDistances(xs, query)
   v('distances', distances)
   tester:asserteq(0, distances[1])
   tester:asserteq(0.2, distances[2])

   -- query is a row in xs
   v('xs', xs)
   v('xs[1]', xs[1])
   distances = kernelSmoother:euclideanDistances(xs, xs[1])
   tester:asserteq(0, distances[1])
   tester:asserteq(0.2, distances[2])
   -- make sure that xs[0] was not mutated
   tester:asserteq(0, xs[1][1])
   tester:asserteq(0, xs[1][2])
   tester:asserteq(0, xs[1][3])
end -- tests._euclideanDistance

function test.weightedAverage()
   local nObs = 3
   local ys = torch.Tensor(nObs)
   ys[1] = 1
   ys[2] = 2
   ys[3] = 3
   
   local weights = torch.Tensor(nObs)
   weights[1] = 0
   weights[2] = 20
   weights[3] = 10
   
   local kwavg = KernelSmoother()
   local tol = 1e-6
   local ok, estimate =  kwavg:weightedAverage(ys, weights)
   tester:assert(ok)
   tester:assert(math.abs(2.333333333 - estimate) < tol)

   weights = torch.Tensor(nObs):fill(0)
   local ok, estimate = kwavg:weightedAverage(ys, weights)
   tester:assert(not ok)
   tester:asserteq(estimate, 'all weights used were 0')
end -- weightedAverage

function test.weights()
   local nObs = 3
   local nDims = 2
   local xs = torch.Tensor(nObs, nDims)
   xs[1] = torch.Tensor(nDims):fill(1)
   xs[2] = torch.Tensor(nDims):fill(2)
   xs[3] = torch.Tensor(nDims):fill(3)

   local kwavg = KernelSmoother()

   local query = torch.Tensor(nDims):fill(0)
   local lambda = 2
   local weights = kwavg:weights(xs, query, lambda)

   local tol = 1e-6
   tester:asserteq(3, weights:size(1))
   tester:assert(math.abs(0.3750 - weights[1]) < tol)
   tester:asserteq(0, weights[2])
   tester:asserteq(0, weights[3])

   query = torch.Tensor(nDims):fill(2)
   weights = kwavg:weights(xs, query, lambda)

   tester:asserteq(3, weights:size(1))
   tester:assert(math.abs(0.3750 - weights[1]) < tol)
   tester:assert(math.abs(0.75 - weights[2]) < tol)
   tester:assert(math.abs(0.3750 - weights[3]) < tol)
end -- weights



tester:add(test)
tester:run(true) -- true ==> verbose