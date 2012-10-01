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



tester:add(test)
tester:run(true) -- true ==> verbose